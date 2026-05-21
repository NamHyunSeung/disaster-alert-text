package com.disaster.classifier

import android.Manifest
import android.app.NotificationManager
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.disaster.classifier.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        NotificationHelper.createChannels(this)
        requestNotificationPermission()

        val prefs = getSharedPreferences("settings", MODE_PRIVATE)
        binding.etServerUrl.setText(prefs.getString("server_url", "https://nhs0327-disaster-classifier.hf.space"))

        binding.btnSave.setOnClickListener {
            val url = binding.etServerUrl.text.toString().trimEnd('/')
            prefs.edit().putString("server_url", url).apply()
            Toast.makeText(this, "저장됨", Toast.LENGTH_SHORT).show()
        }

        binding.btnTest.setOnClickListener {
            val url = binding.etServerUrl.text.toString().trimEnd('/')
            binding.tvStatus.text = "연결 확인 중..."
            CoroutineScope(Dispatchers.IO).launch {
                val ok = ApiClient.health(url)
                withContext(Dispatchers.Main) {
                    binding.tvStatus.text = if (ok) "서버 연결 성공" else "서버 연결 실패"
                }
            }
        }

        binding.btnNotificationAccess.setOnClickListener {
            startActivity(Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS))
        }

        updateListenerStatus()
    }

    override fun onResume() {
        super.onResume()
        updateListenerStatus()
    }

    private fun updateListenerStatus() {
        val enabled = isNotificationListenerEnabled()
        binding.tvListenerStatus.text = if (enabled) "알림 접근 허용됨" else "알림 접근 미허용 (버튼으로 설정 필요)"
    }

    private fun isNotificationListenerEnabled(): Boolean {
        val flat = Settings.Secure.getString(contentResolver, "enabled_notification_listeners") ?: return false
        return flat.contains(packageName)
    }

    private fun requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED
            ) {
                ActivityCompat.requestPermissions(
                    this, arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1
                )
            }
        }
    }
}
