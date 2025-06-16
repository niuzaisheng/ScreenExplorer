import os
import jinja2

from .action_selection import ActionSelection

template_dir = os.path.dirname(__file__)
loader = jinja2.FileSystemLoader(template_dir)
jinja2_env = jinja2.Environment(loader=loader)
