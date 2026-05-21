package com.disaster.classifier

import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

// Samsung One UI CBS 알림을 발송하는 시스템 앱 패키지들
private val CBS_PACKAGES = setOf(
    "com.sec.android.app.sbrowser",
    "com.samsung.android.cbsreceiver",
    "com.android.cellbroadcastreceiver",
    "com.google.android.cellbroadcastreceiver",
)

class AlertNotificationListener : NotificationListenerService() {

    override fun onNotificationPosted(sbn: StatusBarNotification) {
        if (sbn.packageName !in CBS_PACKAGES) return

        val extras = sbn.notification?.extras ?: return
        val text   = extras.getCharSequence("android.text")?.toString()
            ?: extras.getCharSequence("android.bigText")?.toString()
            ?: return

        val prefs     = applicationContext.getSharedPreferences("settings", MODE_PRIVATE)
        val serverUrl = prefs.getString("server_url", "https://nhs0327-disaster-classifier.hf.space") ?: return

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val result = ApiClient.classify(serverUrl, text)
                NotificationHelper.show(applicationContext, text, result)
            } catch (e: Exception) {
                // 서버 오류 시 별도 알림 없음 — 시스템 알림이 이미 표시됨
            }
        }
    }
}
