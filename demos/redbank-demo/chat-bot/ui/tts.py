# Copyright 2025 IBM, Red Hat
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

import openai
import httpx


class TextToSpeech:
    def __init__(self, kokoro_url):
        self.kokoro_url = kokoro_url
        self.output_file = "output.wav"
        self.speaker = "af_sky+af_bella"

    def write_voice(self, text: str):
        try:
            unverified_client = httpx.Client(verify=False)

            client = openai.OpenAI(
                base_url=self.kokoro_url,
                api_key="not-needed",
                http_client=unverified_client,
            )

            with client.audio.speech.with_streaming_response.create(
                model="kokoro", voice=self.speaker, input=text
            ) as response:
                response.stream_to_file(self.output_file)
                return response
        except Exception as e:
            print(e)
