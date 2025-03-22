import os
from groq import Groq
import gradio as gr
from pydub import AudioSegment

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

history = []

def generate_song(lyrics, theme=None, genre=None, artist_style=None, inspired_by=None, language="en"):
    if theme is None:
        theme = "blank"

    messages = [
        {
            "role": "system",
            "content": "The user is a professional lyricist and you need to help them finish their lyrics. The lyrics will be provided to you and you will complete what is already there. If there is no theme, suggest a theme. When giving your response just type the lyrics. No extra text."
        },
        {
            "role": "user",
            "content": f"Theme: {theme}\nGenre: {genre}\nArtist style: {artist_style}\nInspired by: {inspired_by}\nLyrics: {lyrics}\nLanguage: {language}"
        },
    ]

    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=messages,
        temperature=0.6,
        max_completion_tokens=4096,
        top_p=0.95,
        stream=True,
        stop=None,
    )

    response = ""

    for chunk in completion:
        response += chunk.choices[0].delta.content or ""

    result = response.split("</think>", 1)[-1].strip()

    history.append(result)
    
    history_output = "\n\n".join(history)

    return [result, history_output]

with gr.Blocks() as demo:
    gr.Markdown("# AI-Powered Lyrics Finisher")
    lyrics_input = gr.Textbox(label="Lyrics", placeholder="Enter some lyrics...", lines=5)
    theme_input = gr.Textbox(label="Theme", placeholder="Enter a theme...", interactive=True, value="", info="Suggestions: Love, Adventure, Sadness, Motivation, Fantasy")
    genre_input = gr.Textbox(label="Genre", placeholder="Enter a genre...", interactive=True, value="", info="Suggestions: Pop, Rock, Hip-Hop, Jazz, Classical")
    artist_style_input = gr.Textbox(label="Artist Style", placeholder="Enter an artist style...", interactive=True, value="", info="Examples: Taylor Swift, The Beatles, Eminem, Beethoven")
    inspired_by_input = gr.Textbox(label="Inspired By", placeholder="Enter inspirations...", interactive=True, value="", info="Examples: Nature, Heartbreak, Space, History")
    language_input = gr.Dropdown(label="Language", choices=["en", "es", "fr", "de", "pt", "it", "zh"], value="en", info="Choose the language for the lyrics")
    submit_btn = gr.Button("Generate", variant='primary')
    history_output = gr.Textbox(label="History", interactive=False, lines=10)

    submit_btn.click(generate_song, inputs=[lyrics_input, theme_input, genre_input, artist_style_input, inspired_by_input, language_input], outputs=[lyrics_input, history_output])

demo.launch()
