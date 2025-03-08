plugins {
    kotlin("jvm") version "2.1.10"
}

group = "ml.from.scratch"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    // multik - like numpy
    implementation("org.jetbrains.kotlinx:multik-core:0.2.3")
    implementation("org.jetbrains.kotlinx:multik-default:0.2.3") // Native implementation
    // smile - like scikit-learn
    implementation("com.github.haifengl:smile-core:4.3.0")
    // kotlin dl = like tf / keras
    implementation("org.jetbrains.kotlinx:kotlin-deeplearning-tensorflow:0.6.0-alpha-1")
    // krangl - like pandas
    implementation("com.github.holgerbrandl:krangl:0.18.4")

    // unit test
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}
kotlin {
    jvmToolchain(21)
}