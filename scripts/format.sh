#!/bin/sh -ex

# Sort imports one per line, so autoflake can remove unused imports
isort --force-single-line-imports bulk_labeling

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place bulk_labeling --exclude=__init__.py
black bulk_labeling
isort bulk_labeling
