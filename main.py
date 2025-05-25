import os
import base64
import asyncio
import io
import time
import logging
from dotenv import load_dotenv
from discord import Client, Intents, Interaction, File, app_commands, Attachment
from discord.ext import commands
import google.genai as genai
import google.genai.types as types
import requests
from bs4 import BeautifulSoup
import pathlib
from PIL import Image
import aiohttp  # UBAH: Menggunakan aiohttp untuk async HTTP requests
from typing import Optional, Dict, List, Tuple, Union

# UBAH: Setup logging untuk debugging yang lebih baik
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# UBAH: Validasi environment variables
required_env_vars = ['GEMINI_API_KEY', 'DISCORD_TOKEN', 'GOOGLE_API_KEY', 'GOOGLE_CSE_ID']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
    exit(1)

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')

system_prompt = """
Jawab dengan bahasa Indonesia. Pastikan output rapi dan mudah dibaca di Discord menggunakan format Markdown:
- Gunakan # untuk heading besar, ## untuk subheading.
- Gunakan - untuk bullet point pada list.
- Gunakan ** untuk teks tebal, * untuk italic.
- Gunakan ``` untuk blok kode (contoh: ```python).
- Pisahkan paragraf dengan baris kosong.
- Batasi pesan agar tidak melebihi 2000 karakter.
"""

# UBAH: Menggunakan class untuk mengelola state dengan lebih baik
class BotState:
    def __init__(self):
        self.conversation_history: Dict[str, any] = {}
        self.image_history: Dict[str, bytes] = {}
        self.command_cooldowns: Dict[str, float] = {}
        self.channel_activity: Dict[str, bool] = {}
        self.response_mode: Dict[str, str] = {}
        
    def cleanup_old_data(self):
        """Membersihkan data lama untuk menghemat memori"""
        current_time = time.time()
        # Hapus cooldown yang sudah expired
        expired_cooldowns = [key for key, timestamp in self.command_cooldowns.items() 
                           if current_time > timestamp]
        for key in expired_cooldowns:
            del self.command_cooldowns[key]

bot_state = BotState()

MAX_HISTORY = 10
COOLDOWN_TIME = 30  # UBAH: Dalam detik, bukan milliseconds
MAX_FILE_SIZE = 25 * 1024 * 1024  # UBAH: 25MB limit untuk file uploads

SUPPORTED_MIME_TYPES = {
    'image/jpeg': 'image',
    'image/png': 'image',
    'image/gif': 'image',
    'application/pdf': 'pdf',
    'video/mp4': 'video',
    'video/mpeg': 'video',
    'audio/mp3': 'audio',
    'audio/mpeg': 'audio',
    'audio/wav': 'audio',
    'image/jpg': 'image'
}

# UBAH: Session HTTP yang dapat digunakan kembali
http_session: Optional[aiohttp.ClientSession] = None

async def get_http_session() -> aiohttp.ClientSession:
    """Mendapatkan atau membuat HTTP session"""
    global http_session
    if http_session is None or http_session.closed:
        timeout = aiohttp.ClientTimeout(total=30)
        http_session = aiohttp.ClientSession(timeout=timeout)
    return http_session

async def fetch_web_content(url: str) -> str:
    """UBAH: Menggunakan aiohttp dan error handling yang lebih baik"""
    try:
        session = await get_http_session()
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; DiscordBot/1.0)'}
        
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                return f"**Error Scraping**\nHTTP {response.status}: Gagal mengambil konten dari {url}."
            
            html = await response.text()
            
        # UBAH: Parsing HTML tanpa blocking
        soup = BeautifulSoup(html, 'html.parser')
        content = ""
        for elem in soup.select('p, h1, h2, h3, article, main'):
            text = elem.get_text().strip()
            if text:  # UBAH: Hanya tambahkan jika ada teks
                content += text + '\n'
                
        return content[:5000] if content else "Konten tidak ditemukan pada halaman tersebut."
        
    except asyncio.TimeoutError:
        logger.error(f'Timeout error fetching {url}')
        return f"**Error Scraping**\nTimeout saat mengambil konten dari {url}."
    except Exception as error:
        logger.error(f'Error di fetchWebContent: {error}')
        return f"**Error Scraping**\nGagal mengambil konten dari {url}: {str(error)}"

async def translate_text(text: str, target_language: str = 'en') -> str:
    """UBAH: Menggunakan aiohttp dan error handling yang lebih baik"""
    try:
        session = await get_http_session()
        url = f"https://translation.googleapis.com/language/translate/v2?key={GOOGLE_API_KEY}"
        payload = {
            'q': text,
            'target': target_language,
            'format': 'text'
        }
        
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                logger.error(f'Translation API error: {response.status}')
                return text
                
            result = await response.json()
            translated_text = result['data']['translations'][0]['translatedText']
            return translated_text
            
    except Exception as error:
        logger.error(f'Error di translateText: {error}')
        return text

async def google_search(query: str) -> str:
    """UBAH: Menggunakan aiohttp dan error handling yang lebih baik"""
    try:
        session = await get_http_session()
        url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}&q={query}&num=5&lr=lang_id&gl=id"
        
        async with session.get(url) as response:
            if response.status != 200:
                return f"**Error**\nGoogle Search API error: {response.status}"
                
            data = await response.json()
            
        if not data.get('items'):
            return "**Hasil Pencarian**\nMaaf, tidak ada hasil yang ditemukan untuk pencarian ini."

        first_url = data['items'][0]['link']
        web_content = await fetch_web_content(first_url)

        search_results = "**Hasil Pencarian dari Google**\n\n"
        for index, item in enumerate(data['items']):
            title = item.get('title', 'No Title')[:100]  # UBAH: Batasi panjang title
            snippet = item.get('snippet', 'No description')[:200]  # UBAH: Batasi panjang snippet
            search_results += f"- **{index + 1}. {title}**\n"
            search_results += f"  {snippet}\n"
            search_results += f"  Sumber: [Klik di sini]({item['link']})\n\n"

        search_results += f"**Konten dari {first_url}**\n{web_content}\n"
        return search_results
        
    except Exception as error:
        logger.error(f'Error di googleSearch: {error}')
        return "**Error**\nTerjadi kesalahan saat melakukan pencarian Google."

def extract_youtube_url(text: str) -> Optional[str]:
    """UBAH: Menambahkan type hints dan pattern yang lebih robust"""
    import re
    youtube_patterns = [
        r'(https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+)',
        r'(https?://youtu\.be/[\w-]+)',
        r'(https?://(?:www\.)?youtube\.com/embed/[\w-]+)'
    ]
    
    for pattern in youtube_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None

async def generate_response(channel_id: str, prompt: str, media_data: Optional[Dict] = None, 
                          search_query: Optional[str] = None, use_thinking: bool = False, 
                          youtube_url: Optional[str] = None) -> str:
    """UBAH: Error handling yang lebih baik dan timeout protection"""
    try:
        model_name = "gemini-2.0-flash-thinking-exp" if use_thinking else "gemini-2.0-flash"
        
        if channel_id not in bot_state.conversation_history:
            bot_state.conversation_history[channel_id] = client.chats.create(
                model=model_name,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.9,
                    max_output_tokens=4000
                )
            )
        
        chat = bot_state.conversation_history[channel_id]
        contents = [prompt]

        if search_query:
            search_results = await google_search(search_query)
            contents.append(search_results)

        if media_data:
            try:
                if media_data['mime_type'] == 'application/pdf':
                    pdf_buffer = base64.b64decode(media_data['base64'])
                    # UBAH: Validasi ukuran file PDF
                    if len(pdf_buffer) > MAX_FILE_SIZE:
                        return "**Error**\nFile PDF terlalu besar. Maksimal 25MB."
                    pdf_file = client.files.upload(file=io.BytesIO(pdf_buffer), config=dict(mime_type='application/pdf'))
                    contents.append(pdf_file)
                else:
                    file_data = base64.b64decode(media_data['base64'])
                    if len(file_data) > MAX_FILE_SIZE:
                        return "**Error**\nFile terlalu besar. Maksimal 25MB."
                    contents.append(types.Part.from_bytes(
                        data=file_data,
                        mime_type=media_data['mime_type']
                    ))
            except Exception as e:
                logger.error(f'Error processing media: {e}')
                return "**Error**\nGagal memproses file media."

        if youtube_url:
            contents.append(types.Part(
                file_data=types.FileData(file_uri=youtube_url)
            ))

        # UBAH: Menggunakan asyncio.wait_for untuk timeout protection
        try:
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: chat.send_message(contents)
                ),
                timeout=60.0  # 60 detik timeout
            )
        except asyncio.TimeoutError:
            return "**Error**\nTimeout: Permintaan memakan waktu terlalu lama."
        
        response_text = response.text
        if not any(marker in response_text for marker in ['#', '-', '```']):
            paragraphs = [p.strip() for p in response_text.split('\n\n') if p.strip()]
            response_text = '\n\n' + '\n\n'.join(paragraphs)
        
        return response_text
        
    except Exception as error:
        logger.error(f'Error di generateResponse: {error}')
        return f"**Error**\nTerjadi kesalahan saat menghasilkan respons: {str(error)}"

async def enrich_prompt(prompt: str, use_english: bool = False) -> str:
    """UBAH: Error handling dan timeout yang lebih baik"""
    try:
        model_name = "gemini-2.0-flash"
        language_instruction = "Jawab dengan bahasa Indonesia." if not use_english else "Answer in English."
        enrichment_prompt = f"{language_instruction} Bagusin promptnya tanpa mengubah intinya, agar generate gambar bisa lebih bagus (mohon langsung jawabannya saja formatnya): {prompt}"
        
        response = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_name,
                    contents=enrichment_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.9,
                        max_output_tokens=200
                    )
                )
            ),
            timeout=30.0
        )
        
        enriched_prompt = response.candidates[0].content.parts[0].text
        return enriched_prompt.strip()
        
    except asyncio.TimeoutError:
        logger.error('Timeout in enrichPrompt')
        return prompt
    except Exception as error:
        logger.error(f'Error di enrichPrompt: {error}')
        return prompt

async def generate_image(channel_id: str, prompt: str, use_english: bool = False) -> Tuple[Optional[io.BytesIO], str]:
    """UBAH: Error handling dan timeout yang lebih baik"""
    try:
        enriched_prompt = await enrich_prompt(prompt, use_english)
        logger.info(f"Enriched Prompt: {enriched_prompt}")
        
        model_name = "gemini-2.0-flash-exp"
        
        response = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_name,
                    contents=enriched_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=['Text', 'Image'],
                        temperature=0.9
                    )
                )
            ),
            timeout=120.0  # 2 menit timeout untuk image generation
        )
        
        image_data = None
        image_response_text = None
        
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                image_response_text = part.text
            elif part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
                image_data = part.inline_data.data
        
        if image_data:
            bot_state.image_history[channel_id] = image_data
            
            img_buffer = io.BytesIO(image_data)
            img_buffer.seek(0)
            
            response_text = "Gambar Dibuat" if not use_english else "Image Created"
            
            return img_buffer, image_response_text or response_text
        else:
            error_text = "**Error**\nTidak dapat menghasilkan gambar. Silakan coba prompt yang berbeda."
            if use_english:
                error_text = "**Error**\nUnable to generate image. Please try a different prompt."
            return None, error_text
        
    except asyncio.TimeoutError:
        error_text = "**Error**\nTimeout: Pembuatan gambar memakan waktu terlalu lama."
        if use_english:
            error_text = "**Error**\nTimeout: Image generation took too long."
        return None, error_text
    except Exception as error:
        logger.error(f'Error di generateImage: {error}')
        error_text = f"**Error**\nTerjadi kesalahan saat menghasilkan gambar: {str(error)}"
        if use_english:
            error_text = f"**Error**\nAn error occurred while generating the image: {str(error)}"
        return None, error_text

async def edit_image(channel_id: str, prompt: str, image_data: bytes, use_english: bool = False) -> Tuple[Optional[io.BytesIO], str]:
    """UBAH: Error handling dan timeout yang lebih baik"""
    try:
        model_name = "gemini-2.0-flash-exp"
        
        # UBAH: Validasi image data
        try:
            image_buffer = io.BytesIO(image_data)
            pil_image = Image.open(image_buffer)
            pil_image.verify()  # Validasi gambar
        except Exception as e:
            error_text = "**Error**\nGambar tidak valid atau rusak."
            if use_english:
                error_text = "**Error**\nInvalid or corrupted image."
            return None, error_text
        
        response = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_name,
                    contents=[prompt, types.Part.from_bytes(data=image_data, mime_type="image/png")],
                    config=types.GenerateContentConfig(
                        response_modalities=['Text', 'Image'],
                        temperature=0.9
                    )
                )
            ),
            timeout=120.0
        )
        
        edited_image_data = None
        edit_response_text = None
        
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                edit_response_text = part.text
            elif part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
                edited_image_data = part.inline_data.data
        
        if edited_image_data:
            bot_state.image_history[channel_id] = edited_image_data
            
            img_buffer = io.BytesIO(edited_image_data)
            img_buffer.seek(0)
            
            response_text = "Gambar Berhasil Diedit" if not use_english else "Image Successfully Edited"
            
            return img_buffer, edit_response_text or response_text
        else:
            error_text = "**Error**\nTidak dapat mengedit gambar. Silakan coba prompt yang berbeda."
            if use_english:
                error_text = "**Error**\nUnable to edit the image. Please try a different prompt."
            return None, error_text
        
    except asyncio.TimeoutError:
        error_text = "**Error**\nTimeout: Pengeditan gambar memakan waktu terlalu lama."
        if use_english:
            error_text = "**Error**\nTimeout: Image editing took too long."
        return None, error_text
    except Exception as error:
        logger.error(f'Error di editImage: {error}')
        error_text = f"**Error**\nTerjadi kesalahan saat mengedit gambar: {str(error)}"
        if use_english:
            error_text = f"**Error**\nAn error occurred while editing the image: {str(error)}"
        return None, error_text

def split_text(text: str, max_length: int = 1900) -> List[str]:
    """UBAH: Algoritma splitting yang lebih efisien"""
    if len(text) <= max_length:
        return [text]
        
    chunks = []
    current_chunk = ''
    lines = text.split('\n')
    in_code_block = False
    current_language = ''

    for line in lines:
        line_with_newline = line + '\n' if line != lines[-1] else line
        
        # Handle code blocks
        if line.strip().startswith('```'):
            if not in_code_block:
                current_language = line.strip().replace('```', '')
                in_code_block = True
            else:
                in_code_block = False
            
            if len(current_chunk) + len(line_with_newline) > max_length:
                if in_code_block:
                    current_chunk += '\n```'
                chunks.append(current_chunk.strip())
                current_chunk = line_with_newline
            else:
                current_chunk += line_with_newline
            continue

        # Handle long lines
        if len(line) > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ''
            
            # Split long line into parts
            for i in range(0, len(line), max_length):
                part = line[i:i+max_length]
                chunks.append(part)
        else:
            # Normal line handling
            if len(current_chunk) + len(line_with_newline) > max_length:
                if in_code_block:
                    current_chunk += '\n```'
                chunks.append(current_chunk.strip())
                current_chunk = f"```{current_language}\n" if in_code_block else ""
                current_chunk += line_with_newline
            else:
                current_chunk += line_with_newline

    if current_chunk.strip():
        if in_code_block and not current_chunk.endswith('```'):
            current_chunk += '\n```'
        chunks.append(current_chunk.strip())

    return [chunk for chunk in chunks if chunk.strip()]  # UBAH: Hapus chunk kosong

# UBAH: Helper function untuk cooldown management
def check_cooldown(user_id: str, command: str) -> Tuple[bool, float]:
    """Check if user is on cooldown for a command"""
    cooldown_key = f"{user_id}-{command}"
    current_time = time.time()
    cooldown_end_time = bot_state.command_cooldowns.get(cooldown_key, 0)
    
    if current_time < cooldown_end_time:
        remaining_time = cooldown_end_time - current_time
        return True, remaining_time
    
    bot_state.command_cooldowns[cooldown_key] = current_time + COOLDOWN_TIME
    return False, 0

intents = Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    logger.info(f'Bot {bot.user} siap!')
    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} command(s)")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")

# UBAH: Periodic cleanup task
@bot.event
async def on_ready():
    logger.info(f'Bot {bot.user} siap!')
    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} command(s)")
        
        # Start periodic cleanup
        bot.loop.create_task(periodic_cleanup())
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")

async def periodic_cleanup():
    """UBAH: Task pembersihan periodik untuk menghemat memori"""
    while True:
        try:
            await asyncio.sleep(300)  # Cleanup setiap 5 menit
            bot_state.cleanup_old_data()
            logger.info("Performed periodic cleanup")
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")

@bot.tree.command(name="activate", description="Mengaktifkan bot di channel ini")
async def activate(interaction: Interaction):
    """UBAH: Menggunakan helper function untuk cooldown check"""
    user_id = str(interaction.user.id)
    channel_id = str(interaction.channel_id)
    
    on_cooldown, remaining_time = check_cooldown(user_id, "activate")
    if on_cooldown:
        await interaction.response.send_message(
            f"**Cooldown**\nSilakan tunggu {remaining_time:.1f} detik sebelum menggunakan perintah ini lagi.",
            ephemeral=True
        )
        return
    
    bot_state.channel_activity[channel_id] = True
    await interaction.response.send_message("**Status**\nBot diaktifkan di channel ini!")

@bot.tree.command(name="deactivate", description="Menonaktifkan bot di channel ini")
async def deactivate(interaction: Interaction):
    """UBAH: Menggunakan helper function untuk cooldown check"""
    user_id = str(interaction.user.id)
    channel_id = str(interaction.channel_id)
    
    on_cooldown, remaining_time = check_cooldown(user_id, "deactivate")
    if on_cooldown:
        await interaction.response.send_message(
            f"**Cooldown**\nSilakan tunggu {remaining_time:.1f} detik sebelum menggunakan perintah ini lagi.",
            ephemeral=True
        )
        return
    
    bot_state.channel_activity[channel_id] = False
    await interaction.response.send_message("**Status**\nBot dinonaktifkan di channel ini!")

@bot.tree.command(name="pendek", description="Mengaktifkan mode jawaban singkat")
async def pendek(interaction: Interaction):
    """UBAH: Menggunakan helper function untuk cooldown check"""
    user_id = str(interaction.user.id)
    channel_id = str(interaction.channel_id)
    
    on_cooldown, remaining_time = check_cooldown(user_id, "pendek")
    if on_cooldown:
        await interaction.response.send_message(
            f"**Cooldown**\nSilakan tunggu {remaining_time:.1f} detik sebelum menggunakan perintah ini lagi.",
            ephemeral=True
        )
        return
    
    bot_state.response_mode[channel_id] = 'pendek'
    await interaction.response.send_message("**Status**\nMode respons pendek telah diaktifkan di channel ini!")

@bot.tree.command(name="panjang", description="Mengaktifkan mode jawaban panjang")
async def panjang(interaction: Interaction):
    """UBAH: Menggunakan helper function untuk cooldown check"""
    user_id = str(interaction.user.id)
    channel_id = str(interaction.channel_id)
    
    on_cooldown, remaining_time = check_cooldown(user_id, "panjang")
    if on_cooldown:
        await interaction.response.send_message(
            f"**Cooldown**\nSilakan tunggu {remaining_time:.1f} detik sebelum menggunakan perintah ini lagi.",
            ephemeral=True
        )
        return
    
    bot_state.response_mode[channel_id] = 'panjang'
    await interaction.response.send_message("**Status**\nMode respons panjang telah diaktifkan di channel ini!")

# UBAH: Helper function untuk download file
async def download_attachment(attachment: Attachment) -> Optional[bytes]:
    """Download attachment dengan error handling yang baik"""
    try:
        if attachment.size > MAX_FILE_SIZE:
            return None
        
        session = await get_http_session()
        async with session.get(attachment.url) as response:
            if response.status == 200:
                return await response.read()
        return None
    except Exception as e:
        logger.error(f'Error downloading attachment: {e}')
        return None

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    channel_id = str(message.channel.id)
    is_bot_active = bot_state.channel_activity.get(channel_id, False)
    content = message.content.strip()

    # UBAH: Menambahkan rate limiting per user
    user_id = str(message.author.id)
    current_time = time.time()
    user_last_message = bot_state.command_cooldowns.get(f"{user_id}-message", 0)
    
    if current_time - user_last_message < 2:  # 2 detik antara pesan
        return
    
    bot_state.command_cooldowns[f"{user_id}-message"] = current_time

    if content.lower() == '!reset':
        if channel_id in bot_state.conversation_history:
            bot_state.conversation_history.pop(channel_id)
            await message.channel.send('✅ Riwayat percakapan di channel ini telah direset!')
        else:
            await message.channel.send('ℹ️ Tidak ada riwayat percakapan yang perlu dihapus')
        return
    
    if content.lower() == '!resetgambar':
        if channel_id in bot_state.image_history:
            bot_state.image_history.pop(channel_id)
            await message.channel.send('✅ Riwayat gambar di channel ini telah direset!')
        else:
            await message.channel.send('ℹ️ Tidak ada riwayat gambar yang perlu dihapus')
        return

    # UBAH: Improved image generation commands
    if content.lower().startswith('!gambar'):
        async with message.channel.typing():
            image_prompt = content.replace('!gambar', '', 1).strip()
            
            if not image_prompt:
                await message.reply('**Error**\nGunakan format: `!gambar [deskripsi gambar yang diinginkan]`')
                return
            
            try:
                img_buffer, response_text = await generate_image(channel_id, image_prompt, use_english=False)
                
                if img_buffer:
                    await message.channel.send(
                        f"**{response_text}**\n*Prompt: {image_prompt}*", 
                        file=File(fp=img_buffer, filename='generated_image.png')
                    )
                else:
                    await message.channel.send(response_text)
            except Exception as e:
                logger.error(f'Error in gambar command: {e}')
                await message.channel.send("**Error**\nTerjadi kesalahan saat membuat gambar.")
        return
        
    if content.lower().startswith('!gambar_en'):
        async with message.channel.typing():
            image_prompt = content.replace('!gambar_en', '', 1).strip()
            
            if not image_prompt:
                await message.reply('**Error**\nUse format: `!gambar_en [desired image description]`')
                return
            
            try:
                img_buffer, response_text = await generate_image(channel_id, image_prompt, use_english=True)
                
                if img_buffer:
                    await message.channel.send(
                        f"**{response_text}**\n*Prompt: {image_prompt}*", 
                        file=File(fp=img_buffer, filename='generated_image.png')
                    )
                else:
                    await message.channel.send(response_text)
            except Exception as e:
                logger.error(f'Error in gambar_en command: {e}')
                await message.channel.send("**Error**\nAn error occurred while generating the image.")
        return
    
    if content.lower().startswith('!editgambar'):
        async with message.channel.typing():
            edit_prompt = content.replace('!editgambar', '', 1).strip()
            attachment = message.attachments[0] if message.attachments else None
            
            if attachment:
                if not attachment.content_type or not attachment.content_type.startswith('image/'):
                    await message.reply('**Error**\nHanya file gambar yang dapat diedit!')
                    return
                
                # UBAH: Menggunakan helper function untuk download
                image_data = await download_attachment(attachment)
                if image_data is None:
                    await message.reply('**Error**\nGagal mengunduh gambar atau file terlalu besar (maksimal 25MB)!')
                    return
            elif channel_id in bot_state.image_history:
                image_data = bot_state.image_history[channel_id]
            else:
                await message.reply('**Error**\nGunakan format: `!editgambar [deskripsi edit]` dengan melampirkan gambar atau setelah membuat gambar!')
                return
            
            if not edit_prompt:
                await message.reply('**Error**\nGunakan format: `!editgambar [deskripsi edit]` dengan melampirkan gambar!')
                return
            
            try:
                edited_img_buffer, edit_response_text = await edit_image(channel_id, edit_prompt, image_data, use_english=False)
                
                if edited_img_buffer:
                    await message.channel.send(
                        f"**{edit_response_text}**\n*Edit: {edit_prompt}*", 
                        file=File(fp=edited_img_buffer, filename='edited_image.png')
                    )
                else:
                    await message.channel.send(edit_response_text)
            except Exception as e:
                logger.error(f'Error in editgambar command: {e}')
                await message.channel.send("**Error**\nTerjadi kesalahan saat mengedit gambar.")
        return
    
    if content.lower().startswith('!editgambar_en'):
        async with message.channel.typing():
            edit_prompt = content.replace('!editgambar_en', '', 1).strip()
            attachment = message.attachments[0] if message.attachments else None
            
            if attachment:
                if not attachment.content_type or not attachment.content_type.startswith('image/'):
                    await message.reply('**Error**\nOnly image files can be edited!')
                    return
                
                # UBAH: Menggunakan helper function untuk download
                image_data = await download_attachment(attachment)
                if image_data is None:
                    await message.reply('**Error**\nFailed to download image or file too large (max 25MB)!')
                    return
            elif channel_id in bot_state.image_history:
                image_data = bot_state.image_history[channel_id]
            else:
                await message.reply('**Error**\nUse format: `!editgambar_en [edit description]` with an attached image or after generating an image!')
                return
            
            if not edit_prompt:
                await message.reply('**Error**\nUse format: `!editgambar_en [edit description]` with an attached image!')
                return

            try:
                edited_img_buffer, edit_response_text = await edit_image(channel_id, edit_prompt, image_data, use_english=True)
                
                if edited_img_buffer:
                    await message.channel.send(
                        f"**{edit_response_text}**\n*Edit: {edit_prompt}*", 
                        file=File(fp=edited_img_buffer, filename='edited_image.png')
                    )
                else:
                    await message.channel.send(edit_response_text)
            except Exception as e:
                logger.error(f'Error in editgambar_en command: {e}')
                await message.channel.send("**Error**\nAn error occurred while editing the image.")
        return

    if content.lower().startswith('!think'):
        async with message.channel.typing():
            thinking_prompt = content.replace('!think', '', 1).strip()
            
            if not thinking_prompt:
                await message.reply('**Error**\nGunakan format: `!think [pertanyaan atau permintaan]`')
                return
                
            attachment = message.attachments[0] if message.attachments else None
            media_data = None

            if attachment:
                mime_type = attachment.content_type
                if mime_type not in SUPPORTED_MIME_TYPES:
                    supported_formats = ', '.join(set(SUPPORTED_MIME_TYPES.keys()))
                    await message.reply(f'**Error**\nFormat file tidak didukung.\n**Format yang didukung:** {supported_formats}')
                    return

                # UBAH: Menggunakan helper function untuk download
                file_data = await download_attachment(attachment)
                if file_data is None:
                    await message.reply('**Error**\nGagal mengunduh file atau file terlalu besar (maksimal 25MB)!')
                    return
                    
                base64_data = base64.b64encode(file_data).decode('utf-8')
                media_data = {'mime_type': mime_type, 'base64': base64_data}

            try:
                ai_response = await generate_response(channel_id, thinking_prompt, media_data, None, True)
                response_chunks = split_text(ai_response)
                
                for i, chunk in enumerate(response_chunks):
                    await message.channel.send(chunk)
                    # UBAH: Delay yang lebih pendek untuk chunks
                    if i < len(response_chunks) - 1:
                        await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f'Error in think command: {e}')
                await message.channel.send("**Error**\nTerjadi kesalahan saat memproses permintaan thinking.")
        return

    if content.lower().startswith('!gift'):
        async with message.channel.typing():
            gift_prompt = content.replace('!gift', '', 1).strip()
            
            if not gift_prompt:
                await message.reply('**Error**\nGunakan format: `!gift [pertanyaan atau permintaan]`')
                return
                
            try:
                ai_response = await generate_response(channel_id, gift_prompt)
                response_chunks = split_text(ai_response)
                
                for i, chunk in enumerate(response_chunks):
                    await message.channel.send(chunk)
                    # UBAH: Delay yang lebih pendek untuk chunks
                    if i < len(response_chunks) - 1:
                        await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f'Error in gift command: {e}')
                await message.channel.send("**Error**\nTerjadi kesalahan saat memproses permintaan.")
        return

    # UBAH: Main chat logic dengan error handling yang lebih baik
    if is_bot_active or content.startswith('!chat') or content.startswith('!cari'):
        prompt = content
        search_query = None
        youtube_url = extract_youtube_url(content)

        if content.startswith('!cari'):
            search_query = content.replace('!cari', '', 1).strip()
            if not search_query:
                await message.reply('**Error**\nGunakan format: `!cari [kata kunci pencarian]`')
                return
            prompt = f"Berikan jawaban berdasarkan pencarian untuk: {search_query}"
        elif content.startswith('!chat'):
            chat_prompt = content.replace('!chat', '', 1).strip()
            if not chat_prompt:
                await message.reply('**Error**\nGunakan format: `!chat [pertanyaan atau pesan]`')
                return
            prompt = chat_prompt
        elif is_bot_active and not content.startswith('!'):
            prompt = content

        # UBAH: Menyesuaikan prompt berdasarkan mode respons dengan validasi
        mode = bot_state.response_mode.get(channel_id, 'panjang')
        if mode == 'pendek':
            prompt = "Berikan jawaban yang singkat dan padat: " + prompt

        attachment = message.attachments[0] if message.attachments else None
        media_data = None

        if attachment:
            mime_type = attachment.content_type
            if mime_type not in SUPPORTED_MIME_TYPES:
                supported_formats = ', '.join(set(SUPPORTED_MIME_TYPES.keys()))
                await message.reply(f'**Error**\nFormat file tidak didukung.\n**Format yang didukung:** {supported_formats}')
                return

            async with message.channel.typing():
                # UBAH: Menggunakan helper function untuk download
                file_data = await download_attachment(attachment)
                if file_data is None:
                    await message.reply('**Error**\nGagal mengunduh file atau file terlalu besar (maksimal 25MB)!')
                    return
                    
                base64_data = base64.b64encode(file_data).decode('utf-8')
                media_data = {'mime_type': mime_type, 'base64': base64_data}

                try:
                    ai_response = await generate_response(channel_id, prompt, media_data, search_query, youtube_url=youtube_url)
                    response_chunks = split_text(ai_response)
                    
                    for i, chunk in enumerate(response_chunks):
                        await message.channel.send(chunk)
                        # UBAH: Delay yang lebih pendek untuk chunks
                        if i < len(response_chunks) - 1:
                            await asyncio.sleep(0.5)
                except Exception as e:
                    logger.error(f'Error processing message with attachment: {e}')
                    await message.channel.send("**Error**\nTerjadi kesalahan saat memproses pesan dengan lampiran.")
        else:
            async with message.channel.typing():
                try:
                    ai_response = await generate_response(channel_id, prompt, None, search_query, youtube_url=youtube_url)
                    response_chunks = split_text(ai_response)
                    
                    for i, chunk in enumerate(response_chunks):
                        await message.channel.send(chunk)
                        # UBAH: Delay yang lebih pendek untuk chunks
                        if i < len(response_chunks) - 1:
                            await asyncio.sleep(0.5)
                except Exception as e:
                    logger.error(f'Error processing message: {e}')
                    await message.channel.send("**Error**\nTerjadi kesalahan saat memproses pesan.")

    await bot.process_commands(message)

# UBAH: Graceful shutdown handler
@bot.event
async def on_disconnect():
    global http_session
    if http_session and not http_session.closed:
        await http_session.close()
    logger.info("Bot disconnected and cleaned up resources")

# UBAH: Error handler untuk unhandled exceptions
@bot.event
async def on_error(event, *args, **kwargs):
    logger.error(f'Unhandled error in {event}: {args}, {kwargs}', exc_info=True)

# UBAH: Command error handler
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandOnCooldown):
        await ctx.send(f"⏰ Command sedang cooldown. Coba lagi dalam {error.retry_after:.1f} detik.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("❌ Argumen yang diperlukan tidak ditemukan. Periksa format command.")
    else:
        logger.error(f'Command error: {error}', exc_info=True)
        await ctx.send("❌ Terjadi kesalahan saat menjalankan command.")

if __name__ == "__main__":
    try:
        bot.run(os.getenv('DISCORD_TOKEN'))
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        # UBAH: Cleanup on exit
        if http_session and not http_session.closed:
            asyncio.run(http_session.close())