#!/bin/bash

# 定义 Python 文件列表
PYTHON_SCRIPTS=("Main_My.py")

# 定义日志目录
LOG_DIR="logs"
mkdir -p $LOG_DIR  # 创建日志文件夹（如果不存在）

# 定义每个文件运行的次数
RUN_TIMES=10

# 遍历 Python 文件
for script in "${PYTHON_SCRIPTS[@]}"; do
    echo "Starting runs for $script..."
    
    # 遍历运行次数
    for i in $(seq 1 $RUN_TIMES); do
        echo "Running $script (iteration $i)..."
        
        # 为每个运行创建日志文件，并后台运行
        python $script > "$LOG_DIR/${script%.py}_run_$i.log" 2>&1 &
        
        echo "$script (iteration $i) started. Log saved to $LOG_DIR/${script%.py}_run_$i.log"
    done
done

echo "All scripts started. Use 'ps' or 'htop' to monitor progress."