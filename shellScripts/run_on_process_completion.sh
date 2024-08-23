#!/bin/bash

# Define the PID of the process to monitor
TARGET_PID=$1
CUDA_DEVICE_NUMBER=$2
START_THRESHOLD_INDEX=$3
END_THRESHOLD_INDEX=$4
echo "Command line args given : TARGET_PID=${TARGET_PID}, CUDA_DEVICE_NUMBER=${CUDA_DEVICE_NUMBER}, START_THRESHOLD_INDEX=${START_THRESHOLD_INDEX}, END_THRESHOLD_INDEX=${END_THRESHOLD_INDEX}"
SCRIPT_TO_RUN="nohup sh runScriptsDeidentify.sh ${CUDA_DEVICE_NUMBER} ${START_THRESHOLD_INDEX} ${END_THRESHOLD_INDEX} >> ./nohupOuts/nohup_512_deidentify_masked_cuda_device_number_${CUDA_DEVICE_NUMBER}.out 2>&1 &"

# Function to check if the target process is still running
is_process_running() {
    nvidia-smi | grep -w $TARGET_PID > /dev/null
    return $?
}

# Loop until the process is no longer running
while is_process_running; do
    echo "Process $TARGET_PID is still running. Waiting..."
    sleep 60  # Wait for given number of seconds before checking again
done

echo "Process $TARGET_PID has completed."

# Run the Python script
echo "Running the Python script..."
$SCRIPT_TO_RUN
