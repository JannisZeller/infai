from dotenv import load_dotenv

from src.config.factory import get_config
from src.ui.web.adapter import create_web_app

load_dotenv()

app = create_web_app(get_config())
