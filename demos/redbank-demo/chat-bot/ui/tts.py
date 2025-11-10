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
                model="kokoro",
                voice=self.speaker,
                input=text
            ) as response:
                response.stream_to_file(self.output_file)
                return response
        except Exception as e:
            print(e)
