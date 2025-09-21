#include <Arduino.h>

const int candidates[] = {10, 11, 12, 13, 14, 15, 16, 48}; // common S3 LED pins; 48 often on devkits
const int N = sizeof(candidates)/sizeof(candidates[0]);

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("USER APP: start");
  for (int i=0;i<N;i++) {
    pinMode(candidates[i], OUTPUT);
    digitalWrite(candidates[i], LOW); // set known state
    Serial.printf("pin %d set OUTPUT LOW\n", candidates[i]);
  }
}

void loop() {
  // cycle each candidate pin on for 300ms then off
  for (int i=0;i<N;i++) {
    Serial.printf("Toggling pin %d ON\n", candidates[i]);
    digitalWrite(candidates[i], HIGH);
    delay(300);
    digitalWrite(candidates[i], LOW);
    Serial.printf("Toggling pin %d OFF\n", candidates[i]);
    delay(200);
  }
}
