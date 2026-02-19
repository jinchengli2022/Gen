#!/bin/bash
# test_pouring_env.sh
# Quick test script for PouringWater environment

echo "Testing PouringWater Environment Integration"
echo "=============================================="

cd /home/ljc/Git/Gen_VLA_Adapter/gen

echo ""
echo "1. Testing environment import..."
python3 << 'EOF'
try:
    from pouring_water_env import PouringWater
    print("✓ PouringWater class imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)
EOF

echo ""
echo "2. Testing framework integration..."
python3 << 'EOF'
try:
    from robosuite_env import RoboSuiteDataCollector, CUSTOM_ENVS
    print(f"✓ Custom environments registered: {list(CUSTOM_ENVS.keys())}")
    if "PouringWater" in CUSTOM_ENVS:
        print("✓ PouringWater is available in framework")
    else:
        print("✗ PouringWater not found in CUSTOM_ENVS")
except Exception as e:
    print(f"✗ Framework test failed: {e}")
    exit(1)
EOF

echo ""
echo "3. Testing data collection script..."
python3 collect_pouring.py --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ collect_pouring.py script is executable"
else
    echo "✗ collect_pouring.py has issues"
fi

echo ""
echo "=============================================="
echo "Setup complete! Ready to collect data."
echo ""
echo "Quick start commands:"
echo "  python collect_pouring.py --num_episodes 1 --render"
echo "  python simple_collect.py --env_name PouringWater --num_episodes 5"
