import asynciox
import enum
import functools
import re
import time
import traceback
import uuid
import asyncio
from asyncio import open_connection
from typing import Any, Dict

import docker
import requests
import gymnasium
import json
import numpy as np
from gymnasium import spaces

from screen_env.asyncvnc import Client

from pydantic import ValidationError
from schema import ActionSelection

ENV_VERSION = "review"
JSON_BLOCK_PATTERN = re.compile(r"^```(?:json\s+)?(\W.*?)```$", re.DOTALL)

class FormatError(Exception):
    pass

class UnSupportedActionError(Exception):
    pass

class APIError(Exception):
    pass

def async_to_sync(func):
    @functools.wraps(func)
    def sync_func(*args, **kwargs):
        result = asyncio.get_event_loop().run_until_complete(func(*args, **kwargs))
        return result
    return sync_func

def retry_on_timeout(max_retries=3, delay=2):
    """Retry decorator for handling Docker operation timeouts"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout, 
                        requests.exceptions.ConnectionError) as e:
                    retries += 1
                    if retries >= max_retries:
                        print(f"Operation failed after {max_retries} retries due to timeout: {e}")
                        raise
                    print(f"Docker timeout occurred: {e}. Retrying in {delay} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(delay)
            return func(*args, **kwargs)
        return wrapper
    return decorator


class ActionType(enum.Enum):
    none = 0
    move = 1
    click = 2
    right_click = 3
    double_click = 4
    scroll_up = 5
    scroll_down = 6
    drag_to = 7
    key = 8
    text = 9

class MouseButton(enum.Enum):
    left = 0
    middle = 1
    right = 2

class KeyboardModifiers(enum.IntFlag):
    NONE = 0
    SHIFT = 1
    CONTROL = 2
    ALT = 4
    META = 8

# Keyboard keys
vaild_keys = [
    "BackSpace", "Tab", "Linefeed", "Clear", "Return", "Pause", "Scroll_Lock", "Escape", "Delete", "Home", "Left", "Up", "Right", "Down", "Prior", "Page_Up", "Next", "Page_Down", "End", "Begin",
    "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11",
    "space",
    "exclam",  # !
    "quotedbl",  # "
    "numbersign",
    "dollar",  # $
    "percent",  # %
    "ampersand",  # &
    "apostrophe",  # '
    "quoteright",  # '
    "parenleft",  # (
    "parenright",  # )
    "asterisk",  # *
    "plus",  # +
    "comma",  # ,
    "minus",  # -
    "period",  # .
    "slash",  # /
    "colon",  # :
    "semicolon",  # ;
    "less",  # <
    "equal",  # =
    "greater",  # >
    "question",  # ?
    "at",  # @
    "backslash",  # \
    "bracketright",  # ]
    "asciicircum",  # ^
    "underscore",  # _
    "grave",  # `
    "quoteleft",  # `
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
]

def parse_coordinate(coord_str):
    if coord_str is None:
        return None
        
    try:
        # Handle both integer and float formats
        if '.' in coord_str:
            value = float(coord_str)
            # Reject NaN and Inf values
            if not (float('-inf') < value < float('inf')):
                raise FormatError(f"Invalid coordinate value: {coord_str}")
            return value
        else:
            value = int(coord_str)
            # Reasonable range check to prevent integer overflow
            if not (-1000000 <= value <= 1000000):
                raise FormatError(f"Coordinate value out of reasonable range: {coord_str}")
            return value
    except ValueError:
        raise FormatError(f"Invalid coordinate format: {coord_str}")
    

def parse_action(text, mouse_coordinates_type="discrete", video_width=1920, video_height=1080):
    
    # Validate and sanitize input
    if not isinstance(text, str):
        raise FormatError(f"Input must be a string, got {type(text).__name__}")
    
    if len(text) > 256:  # Prevent regex DoS attacks with very long inputs
        raise FormatError(f"Input too long: {len(text)} characters (max: 256)")
    
    if '\n' in text or '\r' in text:
        raise FormatError("Input cannot contain newline characters")
    
    # Normalize whitespace - convert tabs/newlines to spaces and reduce multiple spaces to one
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Define type-specific number patterns based on coordinate type
    if mouse_coordinates_type == "discrete":
        # Only integers for discrete mode: 0 or positive integers without leading zeros
        number_pattern = r'([1-9]\d*|0)'
    elif mouse_coordinates_type == "continuous":
        # Only 0-1 range for continuous mode: 0, 0.x, or 1.0
        number_pattern = r'(0(?:\.\d+)?|1(?:\.0+)?|0?\.\d+)'
    else:
        raise FormatError(f"Invalid mouse_coordinates_type: {mouse_coordinates_type}")
    
    # Handle Text action separately to deal with escaped quotes properly
    text_action_pattern = r'Text\s*\(\s*{0}\s*,\s*{0}\s*,\s*"(.+?)"\)'.format(number_pattern)
    text_match = re.fullmatch(text_action_pattern, text)
    if text_match:
        # Process Text action directly
        x_coord = text_match.group(1)
        y_coord = text_match.group(2)
        text_content = text_match.group(3)
        
        # Parse the coordinates
        mouse_coordinates_x = parse_coordinate(x_coord)
        mouse_coordinates_y = parse_coordinate(y_coord)
        
        # Replace escaped quotes in the text content
        text_content = re.sub(r'\\"', '"', text_content)
        
        # Create and return the action
        return {
            "action_type": ActionType.text.value,
            "discrete_action": [ActionType.text.value, 0],
            "keyboard_modifiers": [0, 0, 0, 0],
            "text_input": text_content,
            "mouse_coordinates_x": mouse_coordinates_x,
            "mouse_coordinates_y": mouse_coordinates_y
        }
    
    # Process the Key action separately to handle modifiers correctly
    key_action_pattern = r'Key\s*\(\s*"([^"]+?)"\s*\)'
    key_match = re.fullmatch(key_action_pattern, text)
    if key_match:
        key_combo = key_match.group(1).split('+')
        
        # Check for reasonable key combination length
        if len(key_combo) > 4:
            raise FormatError(f"Too many keys in combination: {len(key_combo)} (max: 4)")
        
        # Check for empty segments (like "+K" would split into ["", "K"])
        if any(not k.strip() for k in key_combo):
            raise FormatError(f"Invalid key format: Empty modifier or key in '{key_match.group(1)}'")
            
        main_key = key_combo[-1].strip()  # Last one is the main key
        if not main_key:
            raise FormatError("Missing main key")
            
        keyboard_modifiers = [0, 0, 0, 0]  # [shift, ctrl, alt, meta]
        
        # Track used modifiers to detect duplicates
        used_modifiers = set()
        
        # Process modifier keys
        for modifier in key_combo[:-1]:
            modifier = modifier.strip().lower()  # Normalize to lowercase for case-insensitive comparison
            
            # Check for duplicate modifiers
            if modifier in used_modifiers:
                raise FormatError(f"Duplicate modifier key: {modifier}")
            used_modifiers.add(modifier)
            
            if modifier == "shift":
                keyboard_modifiers[0] = 1
            elif modifier in ["control", "ctrl"]:
                keyboard_modifiers[1] = 1
            elif modifier == "alt":
                keyboard_modifiers[2] = 1
            elif modifier in ["meta", "super"]:
                keyboard_modifiers[3] = 1
            else:
                raise FormatError(f"Invalid modifier key: {modifier}")

        if main_key.lower() == "enter":
            main_key = "Return"

        # Set main key
        try:
            # Case-sensitive check for main key
            key_idx = vaild_keys.index(main_key)
        except ValueError:
            # Try case-insensitive match
            main_key_lower = main_key.lower()
            found = False
            for i, valid_key in enumerate(vaild_keys):
                if valid_key.lower() == main_key_lower:
                    key_idx = i
                    found = True
                    break
            if not found:
                raise FormatError(f"Invalid key: {main_key}")
                
        return {
            "action_type": ActionType.key.value,
            "discrete_action": [ActionType.key.value, key_idx],
            "keyboard_modifiers": keyboard_modifiers,
            "text_input": "",
            "mouse_coordinates_x": 0 if mouse_coordinates_type == "discrete" else 0.0,
            "mouse_coordinates_y": 0 if mouse_coordinates_type == "discrete" else 0.0
        }


    # For all other action types, use the regular pattern
    match = re.fullmatch(
        rf"""(?x)  # Enable verbose mode for clarity
        Move\s*\(\s*{number_pattern}\s*,\s*{number_pattern}\s*\) |
        Click\s*\(\s*{number_pattern}\s*,\s*{number_pattern}\s*\) |
        RightClick\s*\(\s*{number_pattern}\s*,\s*{number_pattern}\s*\) |
        DoubleClick\s*\(\s*{number_pattern}\s*,\s*{number_pattern}\s*\) |
        ScrollUp\s*\(\s*{number_pattern}\s*,\s*{number_pattern}\s*\) |
        ScrollDown\s*\(\s*{number_pattern}\s*,\s*{number_pattern}\s*\) |
        DragTo\s*\(\s*{number_pattern}\s*,\s*{number_pattern}\s*\)
        """, 
        text
    )
    
    if not match:
        raise FormatError(f"Invalid action format: '{text}'. Must be exactly one valid action.")
    
    # Extract the matched groups into a list to process
    action = match.groups()
    
    # Initialize variables with appropriate types based on coordinate type
    action_type = None
    key_idx = 0
    if mouse_coordinates_type == "discrete":
        mouse_coordinates_x = 0
        mouse_coordinates_y = 0
    elif mouse_coordinates_type == "continuous":
        mouse_coordinates_x = 0.0
        mouse_coordinates_y = 0.0
    else:
        raise FormatError(f"Invalid mouse_coordinates_type: {mouse_coordinates_type}")

    keyboard_modifiers = [0, 0, 0, 0]  # [shift, ctrl, alt, meta]
    text_input = ""
    
    # Process based on matched action type
    if action[0] is not None and action[1] is not None:  # Move(x, y)
        action_type = ActionType.move.value
        mouse_coordinates_x = parse_coordinate(action[0])
        mouse_coordinates_y = parse_coordinate(action[1])
    elif action[2] is not None and action[3] is not None:  # Click(x, y)
        action_type = ActionType.click.value
        mouse_coordinates_x = parse_coordinate(action[2])
        mouse_coordinates_y = parse_coordinate(action[3])
    elif action[4] is not None and action[5] is not None:  # RightClick(x, y)
        action_type = ActionType.right_click.value
        mouse_coordinates_x = parse_coordinate(action[4])
        mouse_coordinates_y = parse_coordinate(action[5])
    elif action[6] is not None and action[7] is not None:  # DoubleClick(x, y)
        action_type = ActionType.double_click.value
        mouse_coordinates_x = parse_coordinate(action[6])
        mouse_coordinates_y = parse_coordinate(action[7])
    elif action[8] is not None and action[9] is not None:  # ScrollUp(x, y)
        action_type = ActionType.scroll_up.value
        mouse_coordinates_x = parse_coordinate(action[8])
        mouse_coordinates_y = parse_coordinate(action[9])
    elif action[10] is not None and action[11] is not None:  # ScrollDown(x, y)
        action_type = ActionType.scroll_down.value
        mouse_coordinates_x = parse_coordinate(action[10])
        mouse_coordinates_y = parse_coordinate(action[11])
    elif action[12] is not None and action[13] is not None:  # DragTo(x, y)
        action_type = ActionType.drag_to.value
        mouse_coordinates_x = parse_coordinate(action[12])
        mouse_coordinates_y = parse_coordinate(action[13])
    else:
        # This shouldn't happen if the regex matched
        raise FormatError("Invalid action format, unknown action type.")

    # Early validation of coordinate ranges based on the type
    if mouse_coordinates_type == "discrete":
        if not isinstance(mouse_coordinates_x, int):
            mouse_coordinates_x = int(mouse_coordinates_x)  # Convert float to int if needed
        if not isinstance(mouse_coordinates_y, int):
            mouse_coordinates_y = int(mouse_coordinates_y)
            
        if mouse_coordinates_x < 0 or mouse_coordinates_x >= video_width:
            raise FormatError(f"Invalid x coordinate: {mouse_coordinates_x} (valid range: 0-{video_width-1})")
        if mouse_coordinates_y < 0 or mouse_coordinates_y >= video_height:
            raise FormatError(f"Invalid y coordinate: {mouse_coordinates_y} (valid range: 0-{video_height-1})")
    elif mouse_coordinates_type == "continuous":
        if not isinstance(mouse_coordinates_x, float):
            mouse_coordinates_x = float(mouse_coordinates_x)  # Convert int to float if needed
        if not isinstance(mouse_coordinates_y, float):
            mouse_coordinates_y = float(mouse_coordinates_y)
            
        if mouse_coordinates_x < 0.0 or mouse_coordinates_x > 1.0:
            raise FormatError(f"Invalid x coordinate: {mouse_coordinates_x} (valid range: 0.0-1.0)")
        if mouse_coordinates_y < 0.0 or mouse_coordinates_y > 1.0:
            raise FormatError(f"Invalid y coordinate: {mouse_coordinates_y} (valid range: 0.0-1.0)")

    parsed_action = {
        "action_type": action_type,
        "discrete_action": [action_type, key_idx],
        "keyboard_modifiers": keyboard_modifiers,
        "text_input": text_input,
        "mouse_coordinates_x": mouse_coordinates_x,
        "mouse_coordinates_y": mouse_coordinates_y
    }

    return parsed_action


def generate_action_string(action: Dict[str, Any], mouse_coordinates_type="discrete", video_width=1920, video_height=1080):
    discrete_action = action["discrete_action"]
    action_type = ActionType(int(discrete_action[0]))
    if mouse_coordinates_type == "discrete":
        mouse_coordinates_x = int(action["mouse_coordinates_x"])
        mouse_coordinates_y = int(action["mouse_coordinates_y"])
    elif mouse_coordinates_type == "continuous":
        mouse_coordinates_x = int(action["mouse_coordinates_x"] * video_width)
        mouse_coordinates_y = int(action["mouse_coordinates_y"] * video_height)

    keyboard_key = vaild_keys[discrete_action[1]]
    modifiers = action["keyboard_modifiers"]
    text_input = action["text_input"]

    action_description = ""
    if action_type == ActionType.move:
        action_description += f"Move({mouse_coordinates_x}, {mouse_coordinates_y})"
    elif action_type == ActionType.click:
        action_description += f"Click({mouse_coordinates_x}, {mouse_coordinates_y})"
    elif action_type == ActionType.right_click:
        action_description += f"RightClick({mouse_coordinates_x}, {mouse_coordinates_y})"
    elif action_type == ActionType.double_click:
        action_description += f"DoubleClick({mouse_coordinates_x}, {mouse_coordinates_y})"
    elif action_type == ActionType.scroll_up:
        action_description += f"ScrollUp({mouse_coordinates_x}, {mouse_coordinates_y})"
    elif action_type == ActionType.scroll_down:
        action_description += f"ScrollDown({mouse_coordinates_x}, {mouse_coordinates_y})"
    elif action_type == ActionType.drag_to:
        action_description += f"DragTo({mouse_coordinates_x}, {mouse_coordinates_y})"
    elif action_type == ActionType.text:
        action_description += f"Text({mouse_coordinates_x}, {mouse_coordinates_y}, \"{text_input}\")"
    elif action_type == ActionType.key:
        shift_pressed = bool(modifiers[0])
        ctrl_pressed = bool(modifiers[1])
        alt_pressed = bool(modifiers[2])
        meta_pressed = bool(modifiers[3])
        press = []
        if shift_pressed:
            press.append("Shift_L")
        if ctrl_pressed:
            press.append("Control_L")
        if alt_pressed:
            press.append("Alt_L")
        if meta_pressed:
            press.append("Meta_L")
        action_description += f"Key({'+'.join(press + [keyboard_key])})"
    elif action_type == ActionType.none:
        action_description += "None"

    return action_description

def action_dict_covert_to_numpy(action, mouse_coordinates_type="discrete", video_width=1920, video_height=1080):
    """
    Convert action dictionary values to NumPy arrays for compatibility with Gym
    
    Parameters:
        action: Dictionary containing action parameters
        
    Returns:
        Dictionary with the same structure but with values converted to NumPy arrays
    """
    if action is None:
        return None
        
    numpy_action = action.copy()

    # Convert discrete_action to numpy array
    if "action_type" in action:
        numpy_action["action_type"] = np.int32(action["action_type"])
    else:
        numpy_action["action_type"] = np.int32(ActionType.none.value)

    if "discrete_action" in action:
        numpy_action["discrete_action"] = np.array(action["discrete_action"], dtype=np.int32)
    else: # default to ActionType.none
        numpy_action["discrete_action"] = np.array([ActionType.none.value, 0], dtype=np.int32)
    
    # Convert keyboard_modifiers to numpy array
    if "keyboard_modifiers" in action:
        numpy_action["keyboard_modifiers"] = np.array(action["keyboard_modifiers"], dtype=np.int32)
    else:
        numpy_action["keyboard_modifiers"] = np.array([0, 0, 0, 0], dtype=np.int32)
    
    # Convert mouse coordinates to numpy scalars
    if "mouse_coordinates_x" in action.keys():
        if mouse_coordinates_type == "discrete":
            # numpy_action["mouse_coordinates_x"] = np.array(action["mouse_coordinates_x"], dtype=np.int32)
            numpy_action["mouse_coordinates_x"] = np.int32(action["mouse_coordinates_x"])
        elif mouse_coordinates_type == "continuous":
            # numpy_action["mouse_coordinates_x"] = np.array(action["mouse_coordinates_x"], dtype=np.float32)
            numpy_action["mouse_coordinates_x"] = np.float32(action["mouse_coordinates_x"])
    else:
        if mouse_coordinates_type == "discrete":
            numpy_action["mouse_coordinates_x"] = np.int32(0)
        elif mouse_coordinates_type == "continuous":
            numpy_action["mouse_coordinates_x"] = np.float32(0.0)
        
    if "mouse_coordinates_y" in action.keys():
        if mouse_coordinates_type == "discrete":
            numpy_action["mouse_coordinates_y"] = np.int32(action["mouse_coordinates_y"])
        elif mouse_coordinates_type == "continuous":
            numpy_action["mouse_coordinates_y"] = np.float32(action["mouse_coordinates_y"])
    else:
        if mouse_coordinates_type == "discrete":
            numpy_action["mouse_coordinates_y"] = np.int32(0)
        elif mouse_coordinates_type == "continuous":
            numpy_action["mouse_coordinates_y"] = np.float32(0.0)
    
    # Text input remains as string
    return numpy_action

def varify_generated_text(generated_text, mouse_coordinates_type="discrete", video_width=1920, video_height=1080):
    action = None
    action_obj = None
    format_reward = False
    action_string = None
    try:
        try:
            action_obj = ActionSelection.model_validate_json(generated_text)
        except ValidationError as e:
            raise FormatError(f"Invalid JSON format: {e}")

        action = parse_action(action_obj.action, mouse_coordinates_type, video_width, video_height)
        action_string = generate_action_string(action, mouse_coordinates_type, video_width, video_height)
        if action is None:
            raise FormatError("Invalid action string.")
        format_reward = True

        action = action_dict_covert_to_numpy(action, mouse_coordinates_type="discrete", video_width=1920, video_height=1080)
    except FormatError as e:
        pass

    return action, action_obj, format_reward, action_string

def create_empty_action():
    empty_action = {
        "discrete_action": [ActionType.none.value, 0],  # ActionType.none = 0 表示不执行任何动作
        "keyboard_modifiers": [0, 0, 0, 0],
        "text_input": "",
        "mouse_coordinates_x": 0,
        "mouse_coordinates_y": 0,
    }
    return action_dict_covert_to_numpy(empty_action)

class ScreenEnv(gymnasium.Env):
    metadata = {"render_modes": ["rgb_array", "rgba_array"]}

    def __init__(self, config, render_mode="rgb_array", env_rank=0, ocr_func=None):
        super().__init__()

        self.config = config
        self.render_mode = render_mode
        self.env_rank = env_rank
        self.ocr_func = ocr_func

        self.experiment_name = config.get("experiment_name", "ScreenEnv")
        self.image_name = config.get("image_name", "sgccr.ccs.tencentyun.com/screenagent/screenagent:2.0")
        self.video_width = config["video_width"]
        self.video_height = config["video_height"]
        self.resolution = f"{self.video_width}x{self.video_height}"
        self.max_steps = config.get("max_steps", -1)
        self.wait_after_action = config.get("wait_after_action", 0.1)
        self.use_remote_clipboard = config.get("use_remote_clipboard", False)
        self._create_docker_client()


        print(f"[ScreenEnv Rank {self.env_rank}] Using Docker client {self.docker_client}")

        self._vnc_ip = config.get("vnc_ip", "localhost")
        self._vnc_password = uuid.uuid4().hex
        self._clipboard_server_secret_token = uuid.uuid4().hex
        print(f"[ScreenEnv Rank {self.env_rank}] VNC password: {self._vnc_password}")
        print(f"[ScreenEnv Rank {self.env_rank}] Clipboard server secret token: {self._clipboard_server_secret_token}")
        self._vnc_port = None
        self._remote_clipboard_host = None
        self._clipboard_server_port = None

        self.vnc = None
        self.container_id = None

        self.now_screenshot = None
        self.scroll_repeat = 5

        self._step = 0
        self._done = False

        if self.render_mode == "rgba_array":
            image_color_dim = 4
        elif self.render_mode == "rgb_array":
            image_color_dim = 3
        else:
            image_color_dim = 3

        self.observation_space = spaces.Box(0, 255, shape=(self.video_height, self.video_width, image_color_dim), dtype=np.uint8)

        action_space = spaces.Dict({
            "action_type": spaces.Discrete(len(ActionType)),
            "discrete_action": spaces.MultiDiscrete([len(ActionType), len(vaild_keys)]),
            "keyboard_modifiers": spaces.MultiBinary(4),  # Shift, Control, Alt, Meta
            "text_input": spaces.Text(max_length=20)
        })
        
        self.mouse_coordinates_type = config["mouse_coordinates_type"]
        if self.mouse_coordinates_type == "discrete":
            action_space["mouse_coordinates_x"] = spaces.Discrete(self.video_width)
            action_space["mouse_coordinates_y"] = spaces.Discrete(self.video_height)
        elif self.mouse_coordinates_type == "continuous":
            action_space["mouse_coordinates_x"] = spaces.Box(0, 1, shape=(1,), dtype=np.float32)
            action_space["mouse_coordinates_y"] = spaces.Box(0, 1, shape=(1,), dtype=np.float32)

        self.action_space = spaces.Dict(action_space)

    def _create_docker_client(self):
        if self.config.get("docker_tcp_url") is not None:
            self.docker_client = docker.DockerClient(
                base_url=self.config["docker_tcp_url"],
                timeout=self.config.get("docker_timeout", 120)
            )
        else:
            self.docker_client = docker.from_env(timeout=self.config.get("docker_timeout", 120))
        
    async def _get_obs(self):
        assert self.vnc is not None, "VNC connection is not established."
        self.now_screenshot = await self.vnc.screenshot()  # RGBA
        if self.render_mode == "rgb_array":
            self.now_screenshot = self.now_screenshot[:, :, :3] # RGB
        return self.now_screenshot

    def _get_info(self):
        return {
            "video_height": self.video_height,
            "video_width": self.video_width,
            "step": self._step,
        }
    
    @async_to_sync
    def render(self):
        return self._get_obs()

    async def connect_to_vnc(self):
        self._reader, self._writer = await open_connection(self._vnc_ip, self._vnc_port)
        self.vnc = await Client.create(reader=self._reader, writer=self._writer, password=self._vnc_password)
        assert self.vnc.video.height == self.video_height
        assert self.vnc.video.width == self.video_width
        return True

    async def wait_for_container_health_by_io(self, timeout=20):
        start_time = time.time()

        while True:
            try:
                await self.connect_to_vnc()
                return True
            except ValueError as e:
                print(f"Waiting for connection to VNC service on container {self.container_name}...")

            if time.time() - start_time > timeout:
                print(f"Timeout Waiting for connection to VNC service on container {self.container_name}...")
                return False
            time.sleep(1)

    @retry_on_timeout(max_retries=3)
    def create_containers(self):
        try:
            container = self.docker_client.containers.run(
                image=self.image_name,
                name=f"{self.experiment_name}-{uuid.uuid4().hex}",
                environment={"RESOLUTION": self.resolution, "VNC_PASSWORD": self._vnc_password, 
                            "CLIPBOARD_SERVER_SECRET_TOKEN": self._clipboard_server_secret_token},
                ports={5900: None, 8001: None},
                volumes={"/dev/shm": {"bind": "/dev/shm", "mode": "rw"}},
                detach=True
            )
            self.container = container
            print(f"[ScreenEnv Rank {self.env_rank}] Started container {container.name} ({container.id[:12]})")
            self.container_id = container.id
            self.container_name = container.name
            
            max_reload_attempts = 3
            for attempt in range(max_reload_attempts):
                try:
                    container.reload()
                    break
                except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
                    if attempt == max_reload_attempts - 1:
                        print(f"[ScreenEnv Rank {self.env_rank}] Failed to reload container: {e}")
                        raise
                    print(f"[ScreenEnv Rank {self.env_rank}] Container reload timed out, retrying... ({attempt+1}/{max_reload_attempts})")
                    time.sleep(2)

            self._vnc_port = container.attrs['NetworkSettings']['Ports']['5900/tcp'][0]['HostPort']
            self._clipboard_server_port = container.attrs['NetworkSettings']['Ports']['8001/tcp'][0]['HostPort']
            
            print(f"[ScreenEnv Rank {self.env_rank}] Container {container.name} is running on {self._vnc_ip}:{self._vnc_port}")
            print(f"[ScreenEnv Rank {self.env_rank}] Clipboard server is running on {self._vnc_ip}:{self._clipboard_server_port}")
        except Exception as e:
            print(f"[ScreenEnv Rank {self.env_rank}] Container creation error: {e}")
            if hasattr(self, 'container_id') and self.container_id:
                try:
                    self.remove_containers()
                except Exception as cleanup_error:
                    print(f"[ScreenEnv Rank {self.env_rank}] Cleanup error: {cleanup_error}")
            raise


    @retry_on_timeout(max_retries=3)
    def remove_containers(self):
        if self.container_id is not None:
            try:
                container = self.docker_client.containers.get(self.container_id)
                container.remove(force=True)
                print(f"[ScreenEnv Rank {self.env_rank}] Container {self.container_id[:12]} removed.")
            except docker.errors.NotFound:
                print(f"[ScreenEnv Rank {self.env_rank}] Container {self.container_id[:12]} not found (already removed).")
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                print(f"[ScreenEnv Rank {self.env_rank}] Timeout removing container {self.container_id[:12]}: {e}")
            except Exception as e:
                print(f"[ScreenEnv Rank {self.env_rank}] Error removing container {self.container_id[:12]}: {e}")
            finally:
                self.container_id = None

    @async_to_sync
    async def reset(self, **kwargs):
        print(f"[ScreenEnv Rank {self.env_rank}] Resetting ScreenEnv...")

        if self.vnc is not None:
            try:
                await self.vnc.disconnect()
            except Exception as e:
                print(f"[ScreenEnv Rank {self.env_rank}] Error disconnecting VNC: {e}")
                pass

        retry = 0
        retry_limit = 5
        while retry < retry_limit:
            try:
                self.remove_containers()
                self.create_containers()
                break
            except Exception as e:
                print(f"[ScreenEnv Rank {self.env_rank}] Error removing or creating containers: {e}")
                retry += 1
                if retry >= retry_limit:
                    print(f"[ScreenEnv Rank {self.env_rank}] Max retries reached. Exiting.")
                    raise RuntimeError(f"Failed to create container after {retry_limit} attempts.")
        
        time.sleep(5)
        retry = 0
        while not await self.wait_for_container_health_by_io():  # more quick health check
            print(f"[ScreenEnv Rank {self.env_rank}] VNC service on container {self.container_name} failed to start. Retry {retry}")
            self.remove_containers()
            self.create_containers()
            time.sleep(5)
            retry += 1
            if retry > 5:
                raise RuntimeError(f"[ScreenEnv Rank {self.env_rank}] VNC service on container {self.container_name} failed to start.")

        self._step = 0
        observation = await self._get_obs()
        info = self._get_info()
        print(f"[ScreenEnv Rank {self.env_rank}] ScreenEnv reset complete.")
        return observation, info

    def _send_action(self, action: Dict[str, Any]):
        # print(action)
        discrete_action = action["discrete_action"]
        action_type = ActionType(int(discrete_action[0]))
        keyboard_key = vaild_keys[int(discrete_action[1])]
        modifiers = action["keyboard_modifiers"]
        text_input = action["text_input"]
        if self.config["mouse_coordinates_type"] == "discrete":
            mouse_coordinates_x = int(action["mouse_coordinates_x"])
            mouse_coordinates_y = int(action["mouse_coordinates_y"])
        elif self.config["mouse_coordinates_type"] == "continuous":
            mouse_coordinates_x = int(action["mouse_coordinates_x"] * self.video_width)
            mouse_coordinates_y = int(action["mouse_coordinates_y"] * self.video_height)

        # mouse action
        if action_type == ActionType.move:
            self.vnc.mouse.move(mouse_coordinates_x, mouse_coordinates_y)
        elif action_type == ActionType.click:
            self.vnc.mouse.move(mouse_coordinates_x, mouse_coordinates_y)
            self.vnc.mouse.click(MouseButton.left.value)
        elif action_type == ActionType.right_click:
            self.vnc.mouse.move(mouse_coordinates_x, mouse_coordinates_y)
            self.vnc.mouse.click(MouseButton.right.value)
        elif action_type == ActionType.double_click:
            self.vnc.mouse.move(mouse_coordinates_x, mouse_coordinates_y)
            self.vnc.mouse.click(MouseButton.left.value)
            self.vnc.mouse.click(MouseButton.left.value)
        elif action_type == ActionType.scroll_up:
            self.vnc.mouse.move(mouse_coordinates_x, mouse_coordinates_y)
            self.vnc.mouse.scroll_up(repeat=self.scroll_repeat)
        elif action_type == ActionType.scroll_down:
            self.vnc.mouse.move(mouse_coordinates_x, mouse_coordinates_y)
            self.vnc.mouse.scroll_down(repeat=self.scroll_repeat)
        elif action_type == ActionType.drag_to:
            self.vnc.mouse.down(MouseButton.left.value)
            self.vnc.mouse.move(mouse_coordinates_x, mouse_coordinates_y)
            self.vnc.mouse.up(MouseButton.left.value)
        # keyboard action
        elif action_type == ActionType.key:
            shift_pressed = bool(modifiers[0])
            ctrl_pressed = bool(modifiers[1])
            alt_pressed = bool(modifiers[2])
            meta_pressed = bool(modifiers[3])
            press = []
            if shift_pressed:
                press.append("Shift_L")
            if ctrl_pressed:
                press.append("Control_L")
            if alt_pressed:
                press.append("Alt_L")
            if meta_pressed:
                press.append("Meta_L")
            press.append(keyboard_key)
            self.vnc.keyboard.press(*press)
        elif action_type == ActionType.text:
            self.vnc.mouse.move(mouse_coordinates_x, mouse_coordinates_y)
            self.vnc.mouse.click(MouseButton.left.value)     
            if not all(c in vaild_keys for c in text_input):
                if self.use_remote_clipboard:
                    url = f"http://{self._vnc_ip}:{self._clipboard_server_port}/clipboard"
                    data = {
                        "text": text_input,
                        "token": self._clipboard_server_secret_token
                    }
                    try:
                        r = requests.post(url, json=data)
                        print("remote clipboard server response:", r)
                        if r.status_code == 200:
                            self.vnc.keyboard.press('Control_L', 'v')
                    except Exception as e:
                        print(f"[ScreenEnv Rank {self.env_rank}] Remote clipboard error: {e}")
                else:
                    print(f"[ScreenEnv Rank {self.env_rank}] Warning: Text input \"{text_input}\" contains invalid characters. And remote clipboard is not enabled.")
            else:
                self.vnc.keyboard.write(text_input)
        elif action_type == ActionType.none:
            pass

    @async_to_sync
    async def step(self, action):
        reward = 0
        if self._done:
            raise RuntimeError("Cannot step in a done environment.")

        info = self._get_info()
        before_observation = self.now_screenshot
        try:
            self._send_action(action)
            send_time = time.time()  # Start timing before OCR

            self._done = False
            if self.max_steps != -1:
                self._done = (self._step >= self.max_steps)
            terminated = self._done
        
            before_screen_ocr_results=None
            if self.ocr_func is not None:
                before_screen_ocr_results = self.ocr_func(before_observation)[0]
                before_screen_ocr_results = json.dumps(before_screen_ocr_results, ensure_ascii=False)
                info["before_screen_ocr_results"] = before_screen_ocr_results
            
            elapsed_time = time.time() - send_time  # Calculate elapsed time for OCR
            # Wait for the remaining time, if any
            wait_time = self.wait_after_action - elapsed_time
            if wait_time > 0:
                await asyncio.sleep(wait_time)

            after_observation = await self._get_obs()
            self._step += 1

            after_screen_ocr_results=None
            if self.ocr_func is not None:
                after_screen_ocr_results = self.ocr_func(after_observation)[0]
                after_screen_ocr_results = json.dumps(after_screen_ocr_results, ensure_ascii=False)
                info["after_screen_ocr_results"] = after_screen_ocr_results

        except Exception as e:
            print(f"[ScreenEnv Rank {self.env_rank}]  [ScreenEnv.step] Exception", e)
            traceback.print_exc()
            terminated = True

        action_string = generate_action_string(action, self.mouse_coordinates_type, self.video_width, self.video_height)
        info["action_string"] = action_string
        return after_observation, reward, terminated, False, info

    @async_to_sync
    async def close(self):
        if self.vnc is not None:
            try:
                await self.vnc.disconnect()
            except Exception as e:
                print(f"[ScreenEnv Rank {self.env_rank}] [ScreenEnv.close] Failed to disconnect VNC.")
                traceback.print_exc()
            self.vnc = None

        self.remove_containers()

    def parse_action(self, text):
        return parse_action(text, self.mouse_coordinates_type, self.video_width, self.video_height)


action_colors = {
    ActionType.click: "red",
    ActionType.right_click: "orange",
    ActionType.double_click: "purple",
    ActionType.move: "blue",
    ActionType.scroll_up: "green",
    ActionType.scroll_down: "lime",
    ActionType.drag_to: "yellow",
    ActionType.text: "cyan"
}