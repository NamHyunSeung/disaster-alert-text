package com.disaster.classifier

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.telephony.SmsMessage
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class SmsReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        val pdus = intent.extras?.get("pdus") as? Array<*> ?: return
        val format = intent.getStringExtra("format") ?: "3gpp"

        val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)
        val serverUrl = prefs.getString("server_url", "https://nhs0327-disaster-classifier.hf.space") ?: return

        for (pdu in pdus) {
            val sms = SmsMessage.createFromPdu(pdu as ByteArray, format)
            val text = sms.messageBody ?: continue

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
}
