package com.disaster.classifier

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import android.telephony.SmsCbMessage
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class CbsReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        val messages = extractTexts(intent) ?: return
        val prefs     = context.getSharedPreferences("settings", Context.MODE_PRIVATE)
        val serverUrl = prefs.getString("server_url", "https://nhs0327-disaster-classifier.hf.space") ?: return

        for (text in messages) {
            val pending = goAsync()
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    val result = ApiClient.classify(serverUrl, text)
                    NotificationHelper.show(context, text, result)
                } catch (e: Exception) {
                    NotificationHelper.showFallback(context, text)
                } finally {
                    pending.finish()
                }
            }
        }
    }

    private fun extractTexts(intent: Intent): List<String>? {
        val extras = intent.extras ?: return null
        val msgs = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            extras.getParcelableArray("message", SmsCbMessage::class.java)
        } else {
            @Suppress("DEPRECATION")
            extras.getParcelableArray("message")
        } ?: return null
        return msgs.mapNotNull { (it as? SmsCbMessage)?.messageBody }.ifEmpty { null }
    }
}
