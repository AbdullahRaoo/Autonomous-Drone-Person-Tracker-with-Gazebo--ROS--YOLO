#!/bin/bash
# Script to launch the flat city simulation with object detection

# Colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================${NC}"
echo -e "${GREEN}Launching Flat City Simulation${NC}"
echo -e "${BLUE}=====================================${NC}"

# Launch the simulation with the flat terrain
echo -e "${YELLOW}Starting Gazebo with flat city environment...${NC}"
roslaunch iq_sim flat_city.launch &
GAZEBO_PID=$!

# Wait for Gazebo to initialize properly
sleep 10

# Ask if user wants to run object detection
echo -e "${YELLOW}Do you want to run YOLOv5 object detection? (y/n)${NC}"
read -r run_yolo

if [[ $run_yolo == "y" || $run_yolo == "Y" ]]; then
    echo -e "${YELLOW}Starting YOLOv5 object detection...${NC}"
    # Run YOLOv5 detection in a separate terminal
    gnome-terminal -- bash -c "rosrun obj_detect yolov5_ros.py; read -p 'Press enter to close...'"
fi

# Ask if user wants to control model movements
echo -e "${YELLOW}Do you want to control object movements? (y/n)${NC}"
read -r run_movements

if [[ $run_movements == "y" || $run_movements == "Y" ]]; then
    echo -e "${YELLOW}Starting model movement control...${NC}"
    # Run the model movement script
    python3 $(rospack find obj_detect)/move_models.py
fi

# Keep the script running until user decides to exit
echo -e "${YELLOW}Press Ctrl+C to shut down the simulation${NC}"
wait $GAZEBO_PID