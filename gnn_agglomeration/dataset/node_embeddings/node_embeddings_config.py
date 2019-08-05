import configargparse
import configparser
import logging

p = configargparse.ArgParser(
    default_config_files=['config.ini'])


config = p.parse_args()
logging.info(f"\n{p.format_values()}")

# TODO adapt
pw_parser = configparser.ConfigParser()
pw_parser.read(config.db_host)
config.db_host = pw_parser['DEFAULT']['db_host']
