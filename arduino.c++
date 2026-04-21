#include <Servo.h>

Servo miServo;
const int pinServo = 9;

void setup() {
  Serial.begin(9600);
  miServo.attach(pinServo);
  miServo.write(0); // Posición cerrada/base
}

void loop() {
  if (Serial.available() > 0) {
    char material = Serial.read();

    // Verificamos si es uno de nuestros 4 elementos
    if (material == 'A' || material == 'B' || material == 'C' || material == 'O') {
      
      // Ejemplo: Abrir puerta
      miServo.write(90); 
      delay(3000);       // Espera a que caiga el objeto
      miServo.write(0);  // Cierra
      
      // Limpiar el buffer para evitar movimientos repetidos
      while(Serial.available() > 0) Serial.read();
    }
  }
}