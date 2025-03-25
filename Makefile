.PHONY: help

help: ## print this implemented make commands 
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

define PRINT_HELP_PYSCRIPT
import re, sys

print("cuvoxmap")
print("A Lightweight C++ library for robotics.")
print("GPU-accelerated grid state map & translation with ring buffer \n")
print("Supports various types of grid maps, including 2D and 3D")
print("Usage: make <target> <options>")

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT