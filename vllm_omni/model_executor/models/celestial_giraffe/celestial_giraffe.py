# Copyright 2025 vLLM-Omni Contributors
# SPDX-License-Identifier: Apache-2.0
"""
CelestialGiraffe model for vLLM-Omni.

This model extends Gemma3 with CLIP embedding prediction capabilities,
enabling autoregressive image generation through CLIP latent space.
"""

from collections.abc import Iterable, Mapping
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers import CLIPModel
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing import BaseMultiModalProcessor
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

# Default configuration values (matching your HF implementation)
CLIP_EMBED_SIZE: int = 1024
CLIP_AUTOENCODER_EMBED_SIZES: list[int] = [1024, 768, 512]
CLIP_DECODER_EMBED_SIZES: list[int] = [2560, 1024, 768, 512]
CLIP_AUTOENCODER_DROPOUT_PROB: float = 0.1
CLIP_DECODER_DROPOUT_PROB: float = 0.1
CLIP_IMAGE_MODEL_PATH: str = "openai/clip-vit-large-patch14"


# =============================================================================
# Configuration
# =============================================================================


class CelestialGiraffeConfig(Gemma3Config):
    """Configuration class for CelestialGiraffe model.
    
    Extends Gemma3Config with CLIP embedding prediction parameters.
    """
    
    model_type = "celestial_giraffe"
    
    def __init__(
        self,
        boc_token_id: int = 256000,  # Adjust based on your tokenizer
        clip_embed_token_id: int = 256001,  # Adjust based on your tokenizer
        clip_embed_size: int = CLIP_EMBED_SIZE,
        clip_autoencoder_embed_sizes: list[int] | None = None,
        clip_decoder_embed_sizes: list[int] | None = None,
        clip_autoencoder_dropout_prob: float = CLIP_AUTOENCODER_DROPOUT_PROB,
        clip_decoder_dropout_prob: float = CLIP_DECODER_DROPOUT_PROB,
        clip_image_model_path: str = CLIP_IMAGE_MODEL_PATH,
        cache_dir: str | None = None,
        **kwargs,
    ):
        self.boc_token_id = boc_token_id
        self.clip_embed_token_id = clip_embed_token_id
        self.clip_embed_size = clip_embed_size
        self.clip_autoencoder_embed_sizes = (
            clip_autoencoder_embed_sizes or CLIP_AUTOENCODER_EMBED_SIZES
        )
        self.clip_decoder_embed_sizes = (
            clip_decoder_embed_sizes or CLIP_DECODER_EMBED_SIZES
        )
        self.clip_autoencoder_dropout_prob = clip_autoencoder_dropout_prob
        self.clip_decoder_dropout_prob = clip_decoder_dropout_prob
        self.clip_image_model_path = clip_image_model_path
        self.cache_dir = cache_dir
        super().__init__(**kwargs)


# =============================================================================
# Custom Neural Network Components
# =============================================================================


class ImageClipAutoEncoder(nn.Module):
    """Autoencoder for CLIP embeddings.
    
    Compresses CLIP embeddings through a bottleneck and reconstructs them.
    Used for learning a compact representation and for upsampling predicted
    embeddings back to CLIP space.
    """
    
    def __init__(
        self,
        input_size: int,
        embed_sizes: list[int],
        dropout_prob: float,
    ):
        super().__init__()
        
        # Down-sampling layers
        down_sampling_layers: list[nn.Module] = [
            nn.Linear(input_size, embed_sizes[0], bias=True),
            nn.SiLU(),
        ]
        for i in range(1, len(embed_sizes)):
            down_sampling_layers.append(
                nn.Linear(embed_sizes[i - 1], embed_sizes[i], bias=True)
            )
            down_sampling_layers.append(nn.SiLU())
            if i % 2 == 1:
                down_sampling_layers.append(nn.Dropout(dropout_prob))
        self.down_sampler = nn.Sequential(*down_sampling_layers)
        
        # Bottleneck layer
        self.bottleneck = nn.Linear(embed_sizes[-1], embed_sizes[-1], bias=True)
        
        # Up-sampling layers
        up_sampling_layers: list[nn.Module] = []
        for i in range(1, len(embed_sizes))[::-1]:
            up_sampling_layers.append(
                nn.Linear(embed_sizes[i], embed_sizes[i - 1], bias=True)
            )
            up_sampling_layers.append(nn.SiLU())
            if i % 2 == 1:
                up_sampling_layers.append(nn.Dropout(dropout_prob))
        up_sampling_layers.append(nn.Linear(embed_sizes[0], input_size, bias=True))
        up_sampling_layers.append(nn.Tanh())
        self.up_sampler = nn.Sequential(*up_sampling_layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through autoencoder.
        
        Args:
            x: Input CLIP embeddings [batch, clip_embed_size]
            
        Returns:
            Tuple of (bottleneck_embedding, reconstructed_embedding)
        """
        x_down = self.down_sampler(x)
        x_bottleneck = self.bottleneck(x_down)
        x_up = self.up_sampler(x_bottleneck)
        return x_bottleneck, x_up


class ImageClipDecoder(nn.Module):
    """Decoder that projects LLM hidden states to CLIP bottleneck space.
    
    Takes the hidden state from the language model at <boc> token positions
    and projects it to the bottleneck dimension of the autoencoder.
    """
    
    def __init__(
        self,
        embed_sizes: list[int],
        dropout_prob: float,
    ):
        super().__init__()
        
        decoder_layers: list[nn.Module] = []
        for i in range(len(embed_sizes) - 1):
            decoder_layers.append(
                nn.Linear(embed_sizes[i], embed_sizes[i + 1], bias=True)
            )
            decoder_layers.append(nn.SiLU())
            if i % 2 != 0:
                decoder_layers.append(nn.Dropout(dropout_prob))
        
        # Remove last activation
        self.decoder = nn.Sequential(*decoder_layers[:-1])
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode hidden states to CLIP bottleneck space.
        
        Args:
            x: Hidden states from LLM [batch, hidden_size]
            
        Returns:
            Decoded embeddings in bottleneck space [batch, bottleneck_size]
        """
        return self.decoder(x)


# =============================================================================
# Processor Configuration
# =============================================================================


class ClipProcessorConfig:
    """Configuration for CLIP processor tokens and settings.
    
    Matches your HF ClipProcessorConfig.
    """
    # Special tokens for CLIP embedding protocol
    BOC_TOKEN = "<boc>"  # Begin of CLIP
    CLIP_EMBED_TOKEN = "<clip_embed>"  # CLIP embedding placeholder
    EOC_TOKEN = "<eoc>"  # End of CLIP
    
    # Token IDs (will be set from tokenizer)
    BOC_TOKEN_ID = 256000  # Default, override from tokenizer
    CLIP_EMBED_TOKEN_ID = 256001  # Default, override from tokenizer
    
    # CLIP model for image processing
    CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"


# =============================================================================
# Multimodal Processing
# =============================================================================


class CelestialGiraffeProcessingInfo:
    """Processing information for CelestialGiraffe multimodal inputs."""
    
    def __init__(self, ctx):
        self.ctx = ctx
    
    def get_hf_config(self) -> CelestialGiraffeConfig:
        return self.ctx.get_hf_config(CelestialGiraffeConfig)
    
    def get_hf_processor(self, **kwargs):
        """Get the HuggingFace processor."""
        # Return the tokenizer as the base processor
        # The actual image processing is handled separately
        return self.ctx.get_hf_processor(**kwargs)
    
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}
    
    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int]) -> Mapping[str, int]:
        """Get max tokens per multimodal item.
        
        For CelestialGiraffe, each image uses 3 tokens: <boc><clip_embed><eoc>
        """
        return {"image": 3}  # boc + clip_embed + eoc


class CelestialGiraffeMultiModalProcessor(
    BaseMultiModalProcessor[CelestialGiraffeProcessingInfo]
):
    """Process multimodal inputs for CelestialGiraffe.
    
    Handles CLIP embeddings for image conditioning. This processor:
    1. Expands image placeholders to <boc><clip_embed><eoc> sequences
    2. Processes images through CLIP to get pixel_values
    3. Routes pixel_values to the model for CLIP embedding computation
    
    Matches the behavior of your HF MultimodalImageProcessor (CLIP path only).
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # CLIP image processor for 224x224 images
        self._clip_processor = None
        
        # Token configuration
        self.boc_token = ClipProcessorConfig.BOC_TOKEN
        self.clip_embed_token = ClipProcessorConfig.CLIP_EMBED_TOKEN
        self.eoc_token = ClipProcessorConfig.EOC_TOKEN
        self.full_image_sequence = f"{self.boc_token}{self.clip_embed_token}{self.eoc_token}"
    
    @property
    def clip_processor(self):
        """Lazy load CLIP image processor."""
        if self._clip_processor is None:
            from transformers import AutoImageProcessor
            self._clip_processor = AutoImageProcessor.from_pretrained(
                ClipProcessorConfig.CLIP_MODEL_NAME
            )
        return self._clip_processor
    
    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, Any],
        hf_processor_mm_kwargs: Mapping[str, Any],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Configure how multimodal fields are batched.
        
        pixel_values: CLIP-processed images (224x224), one per <boc> token
        pixel_values_clip_embed: Pre-computed CLIP embeddings (optional)
        """
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
            "pixel_values_clip_embed": MultiModalFieldConfig.batched("image"),
        }
    
    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs,
    ):
        """Update prompts to expand image placeholders.
        
        Replaces <start_of_image> or similar tokens with the full
        <boc><clip_embed><eoc> sequence.
        """
        # Prompt updates handled in _call_hf_processor
        return []
    
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, Any],
        mm_kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Process prompt and multimodal data.
        
        Expands image placeholders and processes images through CLIP.
        """
        # Expand image placeholder tokens to full sequence
        # Handle common placeholder tokens
        processed_prompt = prompt
        for placeholder in ["<start_of_image>", "<image>", "[IMG]"]:
            processed_prompt = processed_prompt.replace(
                placeholder, self.full_image_sequence
            )
        
        # Get tokenizer from info
        tokenizer = self.info.get_hf_processor()
        
        # Tokenize the processed prompt
        text_inputs = tokenizer(
            processed_prompt,
            return_tensors="pt",
            **mm_kwargs.get("text_kwargs", {}),
        )
        
        result = dict(text_inputs)
        
        # Process images if present
        images = mm_data.get("image") or mm_data.get("images")
        if images is not None:
            if not isinstance(images, (list, tuple)):
                images = [images]
            
            # Process through CLIP image processor
            clip_inputs = self.clip_processor(
                images,
                return_tensors="pt",
            )
            result["pixel_values"] = clip_inputs["pixel_values"]
        
        # Pass through pre-computed CLIP embeddings if provided
        clip_embeds = mm_data.get("pixel_values_clip_embed")
        if clip_embeds is not None:
            result["pixel_values_clip_embed"] = clip_embeds
        
        return result


class CelestialGiraffeDummyInputsBuilder(
    BaseDummyInputsBuilder[CelestialGiraffeProcessingInfo]
):
    """Build dummy inputs for model profiling."""
    
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        # Create dummy text with full image placeholder sequences
        image_tokens = f"{ClipProcessorConfig.BOC_TOKEN}{ClipProcessorConfig.CLIP_EMBED_TOKEN}{ClipProcessorConfig.EOC_TOKEN}" * num_images
        return f"Describe: {image_tokens} What do you see?"
    
    def get_dummy_mm_data(self, mm_counts: Mapping[str, int]) -> Mapping[str, Any]:
        num_images = mm_counts.get("image", 0)
        if num_images == 0:
            return {}
        
        # Return dummy CLIP pixel values (224x224x3 images)
        # CLIP ViT-L/14 expects 224x224 images
        return {
            "pixel_values": torch.randn(num_images, 3, 224, 224),
        }


# =============================================================================
# Main Model Class
# =============================================================================


@MULTIMODAL_REGISTRY.register_processor(
    CelestialGiraffeMultiModalProcessor,
    info=CelestialGiraffeProcessingInfo,
    dummy_inputs=CelestialGiraffeDummyInputsBuilder,
)
class CelestialGiraffeForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    """vLLM-compatible CelestialGiraffe model.
    
    Extends Gemma3 with autoregressive CLIP embedding prediction for
    image generation. Uses a 3-token protocol:
    - <boc>: Begin of CLIP - triggers CLIP prediction from hidden state
    - <clip_embed>: Placeholder replaced with predicted/input CLIP embedding
    
    The model can:
    1. Accept CLIP embeddings as input (for conditioning on images)
    2. Predict CLIP embeddings during generation (for image generation)
    """
    
    # Weight mapping from HuggingFace checkpoint to vLLM structure
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Map HF model structure to vLLM
            "model.language_model.model.": "language_model.model.",
            "model.language_model.lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.",
            # Keep custom components as-is
            "image_clip_autoencoder.": "image_clip_autoencoder.",
            "image_clip_decoder.": "image_clip_decoder.",
            "image_clip_adapter.": "image_clip_adapter.",
        }
    )
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        
        config = vllm_config.model_config.hf_config
        
        # Handle both CelestialGiraffeConfig and plain Gemma3Config
        if isinstance(config, CelestialGiraffeConfig):
            self.clip_embed_token_id = config.clip_embed_token_id
            self.boc_token_id = config.boc_token_id
            clip_embed_size = config.clip_embed_size
            clip_autoencoder_embed_sizes = config.clip_autoencoder_embed_sizes
            clip_decoder_embed_sizes = config.clip_decoder_embed_sizes
            clip_autoencoder_dropout_prob = config.clip_autoencoder_dropout_prob
            clip_decoder_dropout_prob = config.clip_decoder_dropout_prob
            clip_image_model_path = config.clip_image_model_path
            cache_dir = config.cache_dir
        else:
            # Fallback to defaults or config attributes
            self.clip_embed_token_id = getattr(config, "clip_embed_token_id", 256001)
            self.boc_token_id = getattr(config, "boc_token_id", 256000)
            clip_embed_size = getattr(config, "clip_embed_size", CLIP_EMBED_SIZE)
            clip_autoencoder_embed_sizes = getattr(
                config, "clip_autoencoder_embed_sizes", CLIP_AUTOENCODER_EMBED_SIZES
            )
            clip_decoder_embed_sizes = getattr(
                config, "clip_decoder_embed_sizes", CLIP_DECODER_EMBED_SIZES
            )
            clip_autoencoder_dropout_prob = getattr(
                config, "clip_autoencoder_dropout_prob", CLIP_AUTOENCODER_DROPOUT_PROB
            )
            clip_decoder_dropout_prob = getattr(
                config, "clip_decoder_dropout_prob", CLIP_DECODER_DROPOUT_PROB
            )
            clip_image_model_path = getattr(
                config, "clip_image_model_path", CLIP_IMAGE_MODEL_PATH
            )
            cache_dir = getattr(config, "cache_dir", None)
        
        self.config = config
        
        # Get text config for the language model
        text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
        
        # Initialize the base Gemma3 language model via vLLM registry
        # This reuses vLLM's optimized Gemma implementation
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            hf_config=text_config,
            architectures=["Gemma3ForCausalLM"],
        )
        
        # Get hidden size from language model
        hidden_size = text_config.hidden_size
        
        # CLIP embedding adapter: projects CLIP embeddings to LLM hidden size
        self.image_clip_adapter = nn.Linear(
            clip_embed_size, hidden_size, bias=True
        )
        
        # CLIP autoencoder: for compressing and reconstructing CLIP embeddings
        self.image_clip_autoencoder = ImageClipAutoEncoder(
            input_size=clip_embed_size,
            embed_sizes=clip_autoencoder_embed_sizes,
            dropout_prob=clip_autoencoder_dropout_prob,
        )
        
        # CLIP decoder: projects LLM hidden states to CLIP bottleneck space
        # Input size is LLM hidden size, output goes through autoencoder upsampler
        self.image_clip_decoder = ImageClipDecoder(
            embed_sizes=clip_decoder_embed_sizes,
            dropout_prob=clip_decoder_dropout_prob,
        )
        
        # Load frozen CLIP model for embedding computation (optional, for raw images)
        self._clip_model: Optional[CLIPModel] = None
        self._clip_image_model_path = clip_image_model_path
        self._clip_cache_dir = cache_dir
        
        # Required for pipeline parallelism support
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
    
    @property
    def clip_model(self) -> CLIPModel:
        """Lazy load CLIP model on first access."""
        if self._clip_model is None:
            self._clip_model = CLIPModel.from_pretrained(
                self._clip_image_model_path,
                cache_dir=self._clip_cache_dir,
            )
            # Freeze CLIP model
            for param in self._clip_model.parameters():
                param.requires_grad = False
            # Move to same device as model
            device = next(self.parameters()).device
            self._clip_model = self._clip_model.to(device)
        return self._clip_model
    
    def _compute_clip_embeddings(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CLIP embeddings from pixel values.
        
        Args:
            pixel_values: Images tensor [num_images, 3, 224, 224]
            
        Returns:
            CLIP embeddings [num_images, clip_embed_size]
        """
        device = pixel_values.device
        
        # Use the CLIP model's vision encoder
        with torch.inference_mode():
            clip_outputs = self.clip_model.vision_model(
                pixel_values=pixel_values.to(self.clip_model.device)
            )
            # Get pooled output (CLS token representation)
            clip_embeds = clip_outputs.pooler_output
        
        return clip_embeds.to(device)
    
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Get input embeddings, merging CLIP embeddings at placeholder positions.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len] or [total_tokens]
            multimodal_embeddings: Optional list of CLIP embeddings to inject
            
        Returns:
            Input embeddings tensor
        """
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        
        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0:
            # Find <clip_embed> token positions
            clip_embed_mask = input_ids == self.clip_embed_token_id
            num_clip_positions = clip_embed_mask.sum().item()
            
            if num_clip_positions > 0:
                # Concatenate all CLIP embeddings
                all_clip_embeds = torch.cat(multimodal_embeddings, dim=0)
                
                if all_clip_embeds.shape[0] != num_clip_positions:
                    logger.warning(
                        f"CLIP embedding count mismatch: got {all_clip_embeds.shape[0]} "
                        f"embeddings but {num_clip_positions} <clip_embed> tokens"
                    )
                
                # Project to LLM space and inject
                adapted_embeds = self.image_clip_adapter(
                    all_clip_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                )
                inputs_embeds[clip_embed_mask] = adapted_embeds[:num_clip_positions]
        
        return inputs_embeds
    
    def get_multimodal_embeddings(self, **kwargs) -> Optional[list[torch.Tensor]]:
        """Extract and compute CLIP embeddings from multimodal inputs.
        
        Handles both:
        - pixel_values: Raw images to process through CLIP
        - pixel_values_clip_embed: Pre-computed CLIP embeddings
        
        Args:
            **kwargs: May contain pixel_values or pixel_values_clip_embed
            
        Returns:
            List of CLIP embedding tensors, or None if no images
        """
        clip_embeds_list = []
        
        # Handle pre-computed CLIP embeddings
        pixel_values_clip_embed = kwargs.get("pixel_values_clip_embed")
        if pixel_values_clip_embed is not None:
            if isinstance(pixel_values_clip_embed, torch.Tensor):
                # Single batch of embeddings
                clip_embeds_list.append(pixel_values_clip_embed)
            elif isinstance(pixel_values_clip_embed, (list, tuple)):
                clip_embeds_list.extend(pixel_values_clip_embed)
        
        # Handle raw pixel values - compute CLIP embeddings
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is not None:
            if isinstance(pixel_values, torch.Tensor):
                computed_embeds = self._compute_clip_embeddings(pixel_values)
                clip_embeds_list.append(computed_embeds)
            elif isinstance(pixel_values, (list, tuple)):
                for pv in pixel_values:
                    computed_embeds = self._compute_clip_embeddings(pv)
                    clip_embeds_list.append(computed_embeds)
        
        return clip_embeds_list if clip_embeds_list else None
    
    def _predict_clip_from_hidden(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Predict CLIP embedding from hidden state at <boc> position.
        
        Args:
            hidden_states: LLM hidden states
            input_ids: Input token IDs
            
        Returns:
            Predicted CLIP embedding or None if no <boc> token
        """
        boc_mask = input_ids == self.boc_token_id
        
        if not boc_mask.any():
            return None
        
        # Get hidden state at <boc> position(s)
        boc_hidden = hidden_states[boc_mask]
        
        # Decode to bottleneck space and upsample to CLIP space
        clip_bottleneck = self.image_clip_decoder(boc_hidden)
        predicted_clip = self.image_clip_autoencoder.up_sampler(clip_bottleneck)
        
        return predicted_clip
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        # vllm-omni specific kwargs for per-request state management:
        runtime_additional_information: Optional[list[dict]] = None,
        request_ids: Optional[list[str]] = None,
        request_token_spans: Optional[list[tuple[int, int]]] = None,
        # Multimodal inputs:
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_clip_embed: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[OmniOutput, torch.Tensor, IntermediateTensors]:
        """Forward pass with autoregressive CLIP embedding prediction.
        
        The model:
        1. Processes any input images through CLIP (if pixel_values provided)
        2. Injects CLIP embeddings at <clip_embed> token positions
        3. Injects any pending CLIP embeddings from previous decode steps
        4. Runs the language model forward pass
        5. Predicts CLIP embeddings for any <boc> tokens generated
        6. Returns updates for the next decode step
        
        Args:
            input_ids: Input token IDs
            positions: Position IDs for rotary embeddings
            intermediate_tensors: For pipeline parallelism
            inputs_embeds: Pre-computed input embeddings (optional)
            runtime_additional_information: Per-request state from previous steps
            request_ids: List of request IDs in batch order
            request_token_spans: Token index ranges for each request
            pixel_values: Raw images to process through CLIP [N, 3, 224, 224]
            pixel_values_clip_embed: Pre-computed CLIP embeddings [N, 1024]
            **kwargs: Additional model kwargs
            
        Returns:
            OmniOutput with hidden states and CLIP prediction updates
        """
        device = input_ids.device if input_ids is not None else positions.device
        batch_size = len(request_ids) if request_ids else 1
        
        # Step 1: Get multimodal embeddings (CLIP) if images are provided
        multimodal_embeddings = None
        if pixel_values is not None or pixel_values_clip_embed is not None:
            multimodal_embeddings = self.get_multimodal_embeddings(
                pixel_values=pixel_values,
                pixel_values_clip_embed=pixel_values_clip_embed,
            )
        
        # Step 2: Get input embeddings with multimodal data merged
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids, multimodal_embeddings)
        elif multimodal_embeddings is not None:
            # inputs_embeds provided but we still need to merge multimodal
            clip_embed_mask = input_ids == self.clip_embed_token_id
            if clip_embed_mask.any():
                all_clip_embeds = torch.cat(multimodal_embeddings, dim=0)
                adapted_embeds = self.image_clip_adapter(
                    all_clip_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                )
                num_positions = clip_embed_mask.sum().item()
                inputs_embeds[clip_embed_mask] = adapted_embeds[:num_positions]
        
        # Step 3: Inject pending CLIP embeddings from previous decode steps
        if runtime_additional_information and request_token_spans:
            for req_idx, req_info in enumerate(runtime_additional_information):
                if not req_info:
                    continue
                    
                pending_clip = req_info.get("pending_clip_embeds")
                if pending_clip is None:
                    continue
                
                start, end = request_token_spans[req_idx]
                req_input_ids = input_ids[start:end]
                clip_mask = req_input_ids == self.clip_embed_token_id
                
                if clip_mask.any():
                    # Move to device and adapt
                    pending_clip_gpu = pending_clip.to(device, inputs_embeds.dtype)
                    clip_adapter_embed = self.image_clip_adapter(pending_clip_gpu)
                    
                    # Inject into embeddings
                    num_to_inject = min(clip_mask.sum().item(), clip_adapter_embed.shape[0])
                    inputs_embeds[start:end][clip_mask][:num_to_inject] = (
                        clip_adapter_embed[:num_to_inject]
                    )
        
        # Step 4: Run language model forward pass
        hidden_states = self.language_model.model(
            input_ids=None,  # Using inputs_embeds
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        
        # Handle pipeline parallelism intermediate tensors
        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states
        
        # Step 5: Predict CLIP embeddings for <boc> tokens
        additional_information_updates: list[dict] = []
        
        if request_token_spans:
            for req_idx in range(batch_size):
                start, end = request_token_spans[req_idx]
                req_input_ids = input_ids[start:end]
                req_hidden = hidden_states[start:end]
                
                update: dict[str, Any] = {}
                
                # Check for <boc> tokens
                boc_mask = req_input_ids == self.boc_token_id
                if boc_mask.any():
                    # Get hidden state at last <boc> position
                    boc_hidden = req_hidden[boc_mask][-1:]
                    
                    # Predict CLIP embedding
                    clip_bottleneck = self.image_clip_decoder(boc_hidden)
                    predicted_clip = self.image_clip_autoencoder.up_sampler(clip_bottleneck)
                    
                    # Store for next decode step
                    update["pending_clip_embeds"] = predicted_clip.detach()
                    
                    logger.debug(
                        f"[CelestialGiraffe] Predicted CLIP embedding for request "
                        f"{request_ids[req_idx] if request_ids else req_idx}"
                    )
                
                additional_information_updates.append(update)
        else:
            # Single request or no span info - process entire batch
            predicted_clip = self._predict_clip_from_hidden(hidden_states, input_ids)
            if predicted_clip is not None:
                additional_information_updates = [{"pending_clip_embeds": predicted_clip.detach()}]
        
        # Step 6: Return OmniOutput with hidden states and updates
        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs={
                "additional_information_update": additional_information_updates,
            } if additional_information_updates else None,
        )
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> Optional[torch.Tensor]:
        """Compute output logits from hidden states.
        
        Args:
            hidden_states: Final hidden states from the model
            sampling_metadata: Optional sampling metadata (unused)
            
        Returns:
            Logits tensor
        """
        return self.language_model.compute_logits(hidden_states)
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load model weights with proper mapping.
        
        Handles weight mapping from HuggingFace checkpoint format to
        vLLM's internal structure.
        
        Args:
            weights: Iterable of (name, tensor) pairs
            
        Returns:
            Set of successfully loaded weight names
        """
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["clip_model."],  # Don't load CLIP weights through this
        )
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        
        # Log summary
        try:
            total_bytes = sum(
                p.numel() * p.element_size()
                for p in self.parameters()
                if p is not None
            )
            device = next(self.parameters()).device
            logger.info(
                "[CelestialGiraffe] Loaded %d weights, %.2f MB, device=%s",
                len(loaded),
                total_bytes / (1024**2),
                device,
            )
        except Exception:
            logger.info("[CelestialGiraffe] Loaded %d weights", len(loaded))
        
        return loaded

