package com.disaster.classifier

import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit

data class ClassifyResult(
    val label:      String,
    val confidence: Float,
    val stage:      String,
    val uncertain:  Boolean,
    val probs:      Map<String, Float>? = null,
)

private data class ClassifyRequest(@SerializedName("message") val message: String)

object ApiClient {
    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .build()
    private val gson = Gson()
    private val JSON = "application/json; charset=utf-8".toMediaType()

    fun classify(serverUrl: String, message: String): ClassifyResult {
        val body = gson.toJson(ClassifyRequest(message)).toRequestBody(JSON)
        val request = Request.Builder()
            .url("$serverUrl/classify")
            .post(body)
            .build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw Exception("HTTP ${response.code}")
            val json = response.body?.string() ?: throw Exception("Empty response")
            return gson.fromJson(json, ClassifyResult::class.java)
        }
    }

    fun health(serverUrl: String): Boolean {
        return try {
            val request = Request.Builder().url("$serverUrl/health").get().build()
            client.newCall(request).execute().use { it.isSuccessful }
        } catch (e: Exception) {
            false
        }
    }
}
