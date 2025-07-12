#!/usr/bin/env python3
"""
MuJoCo Viewer for Snake Robot
启动MuJoCo仿真环境查看器，可视化机器人模型
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

def main():
    # 加载机器人模型
    model_path = './robot/scene_up.xml'
    
    print(f"加载模型: {model_path}")
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        print(f"成功加载模型!")
        print(f"关节数量: {model.njnt}")
        print(f"执行器数量: {model.nu}")
        print(f"传感器数量: {model.nsensor}")
        
        # 打印关节名称
        print("\n关节名称:")
        for i in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                print(f"  {i}: {joint_name}")
        
        # 启动交互式查看器
        print("\n启动MuJoCo查看器...")
        print("控制说明:")
        print("- 鼠标左键拖拽: 旋转视角")
        print("- 鼠标右键拖拽: 平移视角") 
        print("- 鼠标滚轮: 缩放")
        print("- 空格键: 暂停/继续仿真")
        print("- Ctrl+R: 重置仿真")
        print("- 按ESC或关闭窗口退出")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # 仿真循环
            start_time = time.time()
            step = 0
            
            while viewer.is_running():
                step_start = time.time()
                
                # 简单的正弦波控制，让机器人动起来
                if model.nu > 0:
                    for i in range(min(model.nu, 9)):  # 最多控制9个关节
                        data.ctrl[i] = 0.3 * np.sin(0.5 * time.time() + i * np.pi / 4)
                
                # 前进仿真一步
                mujoco.mj_step(model, data)
                
                # 同步查看器
                viewer.sync()
                
                # 控制仿真频率 (30 Hz)
                time_until_next_step = 1.0/30.0 - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                
                step += 1
                
                # 每1000步打印一次信息
                if step % 1000 == 0:
                    elapsed = time.time() - start_time
                    print(f"仿真步数: {step}, 运行时间: {elapsed:.1f}s")
    
    except Exception as e:
        print(f"错误: {e}")
        print("请确保:")
        print("1. MuJoCo已正确安装")
        print("2. 模型文件存在: ./robot/scene_up.xml")
        print("3. 模型文件格式正确")
        return

if __name__ == "__main__":
    main()
