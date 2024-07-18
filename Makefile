.PHONY: help clean ros2_debug ros2_release

define PRINT_HELP_PYSCRIPT
import re, sys

print("cuvoxmap")
print("========")
print("A Lightweight C++ library for robotics.")
print("GPU-accelerated grid state map & translate with ring buffer \n")
print("Usage: make <target> <options>")

for line in sys.stdin:
	match = re.match(r'^([0-9a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help: ## print this implemented make commands 
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: ## clean build folder
	rm -rf build install log

ros2_debug: ## build ROS2 basic package 
	colcon build --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Debug

ros2_release: ## build ROS2 basic package 
	colcon build --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release