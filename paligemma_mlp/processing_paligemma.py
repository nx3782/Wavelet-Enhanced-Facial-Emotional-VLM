# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for PaliGemma.
"""

from typing import List, Optional, Union
import torch
import numpy as np

from transformers.feature_extraction_utils import BatchFeature
from image_utils import ImageInput, is_valid_image, make_flat_list_of_images
from transformers.processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
    _validate_images_text_input_order,
)
from transformers.tokenization_utils_base import (
    AddedToken,
    PreTokenizedInput,
    TextInput,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)

IMAGE_TOKEN = "<image>"
LANDMARK_TOKEN = "<landmark>"

EXTRA_TOKENS = [f"<loc{i:0>4}>" for i in range(1024)] + [f"<seg{i:0>3}>" for i in range(128)]


class PaliGemmaTextKwargs(TextKwargs):
    suffix: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]]


class PaliGemmaImagesKwargs(ImagesKwargs):
    do_convert_rgb: Optional[bool]


class PaliGemmaProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: PaliGemmaTextKwargs
    images_kwargs: PaliGemmaImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "data_format": "channels_first",
        },
    }


# Copied from transformers.models.idefics2.processing_idefics2.is_url
def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


# Copied from transformers.models.idefics2.processing_idefics2.is_image_or_image_url
def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


def _is_str_or_image(elem):
    return isinstance(elem, (str)) or is_image_or_image_url(elem)


def build_string_from_input(prompt, bos_token, image_seq_len, image_token, num_images):
    """
    Builds a string from the input prompt and image tokens.
    For example, for the call:
    build_string_from_input(
        prompt="Prefix str"
        bos_token="<s>",
        image_seq_len=3,
        image_token="<im>",
    )
    The output will be:
    "<im><im><im><s>Initial str"
    Args:
        prompt (`List[Union[str, ImageInput]]`): The input prompt.
        bos_token (`str`): The beginning of sentence token.
        image_seq_len (`int`): The length of the image sequence.
        image_token (`str`): The image token.
        num_images (`int`): Number of images in the prompt.
    """
    return f"{image_token * image_seq_len * num_images}{bos_token}{prompt}\n"


class PaliGemmaProcessor(ProcessorMixin):
    r"""
    Constructs a PaliGemma processor which wraps a PaliGemma image processor and a PaliGemma tokenizer into a single processor.

    [`PaliGemmaProcessor`] offers all the functionalities of [`SiglipImageProcessor`] and [`GemmaTokenizerFast`]. See the
    [`~PaliGemmaProcessor.__call__`] and [`~PaliGemmaProcessor.decode`] for more information.

    Args:
        image_processor ([`SiglipImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`GemmaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = ("GemmaTokenizer", "GemmaTokenizerFast")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        landmark_processor=None,
        chat_template=None,
        **kwargs,
    ):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        if not hasattr(image_processor, "image_seq_length"):
            raise ValueError("Image processor is missing an `image_seq_length` attribute.")

        self.image_seq_length = image_processor.image_seq_length

        if not hasattr(tokenizer, "image_token"):
            image_token = AddedToken(IMAGE_TOKEN, normalized=False, special=True)
            tokens_to_add = {"additional_special_tokens": [image_token]}
            tokenizer.add_special_tokens(tokens_to_add)
            self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        else:
            self.image_token_id = tokenizer.image_token_id
        
        if not hasattr(tokenizer, "landmark_token"):
            landmark_token = AddedToken(LANDMARK_TOKEN, normalized=False, special=True)
            tokens_to_add = {"additional_special_tokens": [landmark_token]}
            tokenizer.add_special_tokens(tokens_to_add)
            self.landmark_token_id = tokenizer.convert_tokens_to_ids(LANDMARK_TOKEN)
        else:
            self.landmark_token_id = tokenizer.landmark_token_id

        tokenizer.add_tokens(EXTRA_TOKENS)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        landmarks: Union[torch.Tensor, np.ndarray] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[PaliGemmaProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to GemmaTokenizerFast's [`~GemmaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        SiglipImageProcessor's [`~SiglipImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        The usage for PaliGemma fine-tuning preparation is slightly different than usual. suffix passed are suffixes to
        the prompt in `text`, and will be placed after the prompt. This is because attention is handled differently for
        the prefix and the suffix. For instance,
        ```python
        image = PIL_cow_image
        prompt = "answer en Where is the cow standing?"
        suffix = "on the beach"
        inputs = processor(text=prompt, images=image, suffix=suffix)
        ```
        Here `inputs` will contain the `input_ids` and `token_type_ids` that follow
        ```python
        inputs["input_ids"][:, 256:]
        # tensor([[     2,   6006,    603,    573,  13910,   9980, 235336,    108,    477,   573,   8318]])
        inputs["token_type_ids"][:, 256:]
        tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
        ```
        Meaning the last three tokens are of "label" ("suffix") type while the other ones are of "prefix" type.


        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
            suffix (`str`, `List[str]`, `List[List[str]]`):
                The suffixes or batch of suffixes to be encoded. Only necessary for finetuning. See https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md
                for more information. If your prompt is "<image> What is on the image", the suffix corresponds to the expected prediction "a cow sitting on a bench".

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`. If `suffix`
              is provided, the `input_ids` will also contain the suffix input ids.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **labels** -- Labels compatible with training if `suffix` is not None
        """
        # check if images and text inputs are reversed for BC
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
            PaliGemmaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        # Suffix is appended as "labels" if provided
        suffix = output_kwargs["text_kwargs"].pop("suffix", None)
        return_token_type_ids = True if suffix is not None else False

        if images is None:
            raise ValueError("`images` must be provided to PaliGemmaProcessor.__call__.")

        if text is None:
            logger.warning_once(
                "You are calling PaliGemmaProcessor with no text prefix; it will behave as an image captioning model."
            )
            text = ""

        # Ensure text is a list of strings
        if _is_str_or_image(text):
            text = [text]
        elif isinstance(text, list) and _is_str_or_image(text[0]):
            pass
        else:
            raise ValueError(
                "`text` must be a string or list of strings."
            )

        # We'll do expansions on this variable
        input_strings = text[:]

        # -----------------------------
        # 1) <image> expansions  (UNCHANGED)
        # -----------------------------
        # if images is not None:
        #     if not any(IMAGE_TOKEN in s for s in input_strings):
        #         # logger.warning(
        #         #     "No `<image>` tokens found in `text`, but images provided. We'll auto-insert `<image>` for you."
        #         # )
        #         # Flatten nested images
        #         if isinstance(text, list) and isinstance(images, list):
        #             if len(images) != len(text):
        #                 raise ValueError(
        #                     f"Mismatch: {len(images)} images vs {len(text)} text prompts. Must match, or structure images as nested lists."
        #                 )
        #         if is_valid_image(images):
        #             images = [[images]]
        #         elif isinstance(images, list) and is_valid_image(images[0]):
        #             images = [[img] for img in images]
        #         elif not (isinstance(images, list) and isinstance(images[0], list) and is_valid_image(images[0][0])):
        #             raise ValueError(
        #                 "Images must be a single image, list of images, or list-of-lists of images."
        #             )

        #         # For each prompt, add self.image_seq_length * (#images) `<image>` tokens, plus <bos>, plus the prompt
        #         new_strings = []
        #         for prompt, image_list in zip(input_strings, images):
        #             n_imgs = len(image_list) if isinstance(image_list, list) else 1
        #             repeated_block = IMAGE_TOKEN * (self.image_seq_length * n_imgs)
        #             new_strings.append(f"{repeated_block}{self.tokenizer.bos_token}{prompt}\n")
        #         input_strings = new_strings
        #         # Flatten images so there's 1:1 match with expansions
        #         images = make_flat_list_of_images(images)

        #     else:
        #         # The original "image stuff" expansions with <bos> inserted right after last <image> block:
        #         expanded_list = []
        #         for sample in input_strings:
        #             expanded = sample.replace(IMAGE_TOKEN, IMAGE_TOKEN * self.image_seq_length)
        #             bos_rfind_index = expanded.rfind(IMAGE_TOKEN)
        #             bos_index = bos_rfind_index + len(IMAGE_TOKEN) if bos_rfind_index != -1 else 0
        #             expanded = (
        #                 expanded[:bos_index] + self.tokenizer.bos_token + expanded[bos_index:]
        #             )
        #             expanded_list.append(expanded + "\n")
        #         input_strings = expanded_list

        # # -----------------------------
        # # 2) <landmark> expansions + reorder <bos>
        # # -----------------------------
        # if landmarks is not None:
        #     # If user never typed <landmark>, we do a single block of 10 at the front
        #     print("landmarks provided, expanding...")
        #     if not any(LANDMARK_TOKEN in s for s in input_strings):
        #         repeated_10 = LANDMARK_TOKEN * 10  # e.g. "<landmark><landmark>...<landmark>"
        #         new_list = []
        #         for s in input_strings:
        #             # Simply prepend the landmarks at the start
        #             new_str = f"{repeated_10}{s}"
        #             if not new_str.endswith("\n"):
        #                 new_str += "\n"
        #             new_list.append(new_str)
        #         input_strings = new_list
        # else:
        #     print("No landmarks provided, skipping landmark expansion.")
        if images is not None:
            if not any(IMAGE_TOKEN in s for s in input_strings):
                # Flatten nested images
                if isinstance(text, list) and isinstance(images, list):
                    if len(images) != len(text):
                        raise ValueError(
                            f"Mismatch: {len(images)} images vs {len(text)} text prompts. Must match, or structure images as nested lists."
                        )
                if is_valid_image(images):
                    images = [[images]]
                elif isinstance(images, list) and is_valid_image(images[0]):
                    images = [[img] for img in images]
                elif not (isinstance(images, list) and isinstance(images[0], list) and is_valid_image(images[0][0])):
                    raise ValueError(
                        "Images must be a single image, list of images, or list-of-lists of images."
                    )

                # Store the original prompts separately so we can add them at the end
                original_prompts = input_strings.copy()
                
                # First, just create strings with image tokens
                new_strings = []
                for image_list in images:
                    n_imgs = len(image_list) if isinstance(image_list, list) else 1
                    repeated_block = IMAGE_TOKEN * (self.image_seq_length * n_imgs)
                    new_strings.append(repeated_block)
                input_strings = new_strings
                # Flatten images so there's 1:1 match with expansions
                images = make_flat_list_of_images(images)
            else:
                # Replace IMAGE_TOKEN with the correct number of tokens
                expanded_list = []
                original_prompts = []
                for sample in input_strings:
                    # Split sample at the last IMAGE_TOKEN to separate prompt
                    last_image_index = sample.rfind(IMAGE_TOKEN) + len(IMAGE_TOKEN)
                    image_part = sample[:last_image_index].replace(IMAGE_TOKEN, IMAGE_TOKEN * self.image_seq_length)
                    prompt_part = sample[last_image_index:]
                    
                    expanded_list.append(image_part)
                    original_prompts.append(prompt_part)
                input_strings = expanded_list

        # -----------------------------
        # 2) <landmark> expansions + add <bos> + original prompt
        # -----------------------------
        if landmarks is not None:
            # Landmarks is a tensor of shape (batch, 10, 478, 3)
            # Get batch size from the landmarks tensor
            batch_size = landmarks.shape[0]
            
            # Ensure the number of landmark batches matches the number of text prompts
            if batch_size != len(input_strings):
                raise ValueError(
                    f"Mismatch: {batch_size} landmark batches vs {len(input_strings)} text prompts. Must match."
                )
            
            # For each prompt, add landmark tokens after image tokens
            new_strings = []
            for i, string in enumerate(input_strings):
                # Each landmark in the batch gets 10 tokens (matching the dimension in the tensor)
                landmark_seq_length = 10
                repeated_block = LANDMARK_TOKEN * landmark_seq_length
                
                # Add landmarks after images
                new_string = f"{string}{repeated_block}"
                new_strings.append(new_string)
            
            input_strings = new_strings

        # Finally, add BOS token and the original prompt
        final_strings = []
        for i, string in enumerate(input_strings):
            # Add BOS token after images and landmarks, then add the original prompt
            final_string = f"{string}{self.tokenizer.bos_token}{original_prompts[i]}"
            if not final_string.endswith("\n"):
                final_string += "\n"
            final_strings.append(final_string)

        input_strings = final_strings

        # 3) If user gave a suffix, we treat it as the "label" portion
        if suffix is not None:
            if _is_str_or_image(suffix):
                suffix = [suffix]
            # always append <eos> to each suffix
            suffix = [sf + self.tokenizer.eos_token for sf in suffix]

        # 4) Convert images to pixel_values
        pixel_values = self.image_processor(images, **output_kwargs["images_kwargs"])["pixel_values"]

        # If max_length is provided for text, we add self.image_seq_length for expansions
        if output_kwargs["text_kwargs"].get("max_length", None) is not None:
            output_kwargs["text_kwargs"]["max_length"] += self.image_seq_length

        # 5) Now we tokenize the fully expanded input_strings
        inputs = self.tokenizer(
            input_strings,
            text_pair=suffix,
            return_token_type_ids=return_token_type_ids,
            **output_kwargs["text_kwargs"],
        )

        return_data = {
            **inputs,
            "pixel_values": pixel_values if images is not None else None,
            "landmark_values": landmarks if landmarks is not None else None,
        }

        # If suffix is provided, build `labels` by ignoring prefix tokens (0) w/ -100
        if return_token_type_ids:
            labels = inputs["input_ids"].masked_fill(inputs["token_type_ids"] == 0, -100)
            return_data.update({"labels": labels})

        return BatchFeature(data=return_data)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Gemma
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Gemma
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names with CLIP->PaliGemma
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["PaliGemmaProcessor"]
