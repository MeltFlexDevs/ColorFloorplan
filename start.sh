#!/bin/env bash

gunicorn -c gunicorn_config.py src.main:application
