plugins{
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}
android{
    namespace = "com.tencent.yolov8ncnn"
    compileSdk = 34
    ndkVersion = "26.2.11394342"

    defaultConfig{
        applicationId = "com.tencent.yolov8ncnn"
        minSdk = 24
        targetSdk = 34
    }

    externalNativeBuild{
        cmake{
            path = file("src/main/jni/CMakeLists.txt")
            version = "3.30.5"
        }
    }
}
dependencies{
    implementation("com.android.support:support-v4:24.0.0")
}
