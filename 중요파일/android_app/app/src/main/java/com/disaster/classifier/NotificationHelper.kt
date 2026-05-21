package com.disaster.classifier

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import androidx.core.app.NotificationCompat

private const val CH_EMERG   = "ch_긴급"
private const val CH_CAUTION = "ch_주의"
private const val CH_NORMAL  = "ch_일반"

object NotificationHelper {

    fun createChannels(ctx: Context) {
        val nm = ctx.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(
            NotificationChannel(CH_EMERG, "긴급 재난문자", NotificationManager.IMPORTANCE_HIGH).apply {
                description = "즉각 대응 필요"
            }
        )
        nm.createNotificationChannel(
            NotificationChannel(CH_CAUTION, "주의 재난문자", NotificationManager.IMPORTANCE_DEFAULT).apply {
                description = "주의가 필요한 재난문자"
            }
        )
        nm.createNotificationChannel(
            NotificationChannel(CH_NORMAL, "일반 재난문자", NotificationManager.IMPORTANCE_LOW).apply {
                description = "정보성 재난문자"
            }
        )
    }

    fun show(ctx: Context, message: String, result: ClassifyResult) {
        val (channelId, title, priority) = when (result.label) {
            "긴급" -> Triple(CH_EMERG,   "[긴급] 재난문자", NotificationCompat.PRIORITY_HIGH)
            "주의" -> Triple(CH_CAUTION, "[주의] 재난문자", NotificationCompat.PRIORITY_DEFAULT)
            else  -> Triple(CH_NORMAL,  "[일반] 재난문자", NotificationCompat.PRIORITY_LOW)
        }

        val conf    = "%.0f%%".format(result.confidence * 100)
        val warn    = if (result.uncertain) " ⚠ 수동 확인 권고" else ""
        val subText = "${result.label} $conf$warn"

        val notification = NotificationCompat.Builder(ctx, channelId)
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setContentTitle(title)
            .setContentText(message)
            .setSubText(subText)
            .setStyle(NotificationCompat.BigTextStyle().bigText(message).setSummaryText(subText))
            .setPriority(priority)
            .setAutoCancel(true)
            .build()

        val nm = ctx.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(message.hashCode(), notification)
    }

    fun showFallback(ctx: Context, message: String) {
        val notification = NotificationCompat.Builder(ctx, CH_NORMAL)
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setContentTitle("[재난문자] 분류 실패")
            .setContentText(message)
            .setStyle(NotificationCompat.BigTextStyle().bigText(message))
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setAutoCancel(true)
            .build()

        val nm = ctx.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(message.hashCode(), notification)
    }
}
