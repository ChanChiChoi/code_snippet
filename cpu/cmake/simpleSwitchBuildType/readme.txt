cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Release
# 下面的多版本构建只对vs xcode有用
cmake -H. -Bbuild -DCMAKE_CONFIGURATION_TYPES="Debug;Release"

