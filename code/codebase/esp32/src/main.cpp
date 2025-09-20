#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>

const char* AP_SSID = "ESP32-S3-TEST";
const char* AP_PASS = "esp32test";

const int LED_PIN = 10;     // LED pin
const int BUTTON_PIN = 21;  // Button pin

volatile uint32_t buttonCount = 0;
bool ledState = false;

WebServer server(80);

portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;

void IRAM_ATTR button_isr() {
  portENTER_CRITICAL_ISR(&mux);
  buttonCount++;
  portEXIT_CRITICAL_ISR(&mux);
}

String pageHtml() {
  String s = "<!doctype html><html><meta name='viewport' content='width=device-width'>";
  s += "<h3>ESP32-S3 LED Test</h3>";
  s += "<div>Button presses: <span id='cnt'>0</span></div>";
  s += "<div>LED: <span id='led'>OFF</span></div>";
  s += "<button onclick=\"fetch('/toggle')\">Toggle LED</button>";
  s += "<script>setInterval(()=>{fetch('/status').then(r=>r.json()).then(j=>{document.getElementById('cnt').innerText=j.count;document.getElementById('led').innerText=j.led?'ON':'OFF';})},500);</script>";
  s += "</html>";
  return s;
}

void handleRoot() { server.send(200, "text/html", pageHtml()); }

void handleToggle() {
  ledState = !ledState;
  digitalWrite(LED_PIN, ledState ? HIGH : LOW);
  server.send(200, "text/plain", ledState ? "1" : "0");
}

void handleStatus() {
  String json = "{\"count\":" + String(buttonCount) + ",\"led\":" + String(ledState ? 1 : 0) + "}";
  server.send(200, "application/json", json);
}

void setup() {
  Serial.begin(115200);
  delay(100);

  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), button_isr, FALLING);

  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);
  Serial.print("AP IP: ");
  Serial.println(WiFi.softAPIP());

  server.on("/", handleRoot);
  server.on("/toggle", handleToggle);
  server.on("/status", handleStatus);
  server.begin();
}

void loop() {
  server.handleClient();
}
