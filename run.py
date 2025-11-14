"""
摄像头目标检测与测距系统 - 统一启动器
支持单路、双路、多路模式
"""
import sys
import os


def print_banner():
    """打印系统横幅"""
    print("=" * 70)
    print("      摄像头目标检测与测距系统 v1.2.0")
    print("=" * 70)
    print()


def print_menu():
    """打印菜单"""
    print("请选择运行模式:")
    print()
    print("  1. 单路模式 - 处理单个摄像头")
    print("  2. 双路模式 - 处理两个摄像头（原main.py）")
    print("  3. 多路模式 - 处理任意数量摄像头（1-N路）")
    print("  4. 退出")
    print()


def get_user_choice():
    """获取用户选择"""
    while True:
        try:
            choice = input("请输入选项 (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return choice
            else:
                print("无效选项，请重新输入")
        except KeyboardInterrupt:
            print("\n\n用户取消")
            return '4'


def run_single_mode():
    """运行单路模式"""
    print("\n--- 单路模式 ---")
    print("从配置文件中选择一个摄像头进行检测")
    print()
    
    # 选择摄像头
    while True:
        try:
            cam_index = input("请选择摄像头索引 (0 或 1，默认 0): ").strip()
            if cam_index == '':
                cam_index = '0'
            if cam_index in ['0', '1']:
                break
            else:
                print("请输入 0 或 1")
        except KeyboardInterrupt:
            return
    
    print(f"\n启动单路模式，使用摄像头 {cam_index}...\n")
    
    # 导入并运行
    try:
        import main_single
        sys.argv = ['main_single.py', '--camera', cam_index]
        main_single.main()
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()


def run_dual_mode():
    """运行双路模式"""
    print("\n--- 双路模式 ---")
    print("同时处理配置文件中的两个摄像头")
    print()
    
    print("启动双路模式...\n")
    
    # 导入并运行
    try:
        import main
        main.main()
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()


def run_multi_mode():
    """运行多路模式"""
    print("\n--- 多路模式 ---")
    print("可以处理任意数量的摄像头")
    print()
    
    print("选项:")
    print("  1. 使用配置文件中的所有摄像头")
    print("  2. 指定要使用的摄像头")
    print()
    
    choice = input("请选择 (1-2，默认 1): ").strip()
    if choice == '':
        choice = '1'
    
    if choice == '1':
        print("\n启动多路模式，使用所有摄像头...\n")
        try:
            import main_multi
            main_multi.main()
        except Exception as e:
            print(f"运行出错: {e}")
            import traceback
            traceback.print_exc()
    
    elif choice == '2':
        cameras = input("请输入摄像头索引（用逗号分隔，如: 0,1 或 0,1,2,3): ").strip()
        print(f"\n启动多路模式，使用摄像头: {cameras}...\n")
        try:
            import main_multi
            sys.argv = ['main_multi.py', '--cameras', cameras]
            main_multi.main()
        except Exception as e:
            print(f"运行出错: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    print_banner()
    
    # 检查配置文件
    config_file = 'config/dual_camera_config_backup_20251110_145700.json'
    if not os.path.exists(config_file):
        print(f"错误: 配置文件不存在: {config_file}")
        print("请确保配置文件在正确的位置")
        return
    
    while True:
        print_menu()
        choice = get_user_choice()
        
        if choice == '1':
            run_single_mode()
            break
        elif choice == '2':
            run_dual_mode()
            break
        elif choice == '3':
            run_multi_mode()
            break
        elif choice == '4':
            print("\n再见！")
            break


if __name__ == '__main__':
    main()