[app]
title = NeuroPocket
package.name = neuropocket
package.domain = org.localai

source.dir = .
source.include_exts = py,png,jpg,kv,atlas

version = 0.1
requirements = python3,kivy==2.3.0

orientation = portrait
fullscreen = 0

# Android specific
android.accept_sdk_license = True
android.permissions = INTERNET
android.api = 33
android.minapi = 21
android.ndk = 25b
android.archs = arm64-v8a, armeabi-v7a
android.allow_backup = True

[buildozer]
log_level = 2
warn_on_root = 1
