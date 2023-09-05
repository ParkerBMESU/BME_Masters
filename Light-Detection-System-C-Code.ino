#include <Wire.h>
#include "SparkFun_AS7341X_Arduino_Library.h"
#include <BH1750.h>
#include "DEV_Config.h"
#include "TSL2591.h"

#define TSL_SENSOR 1
#define BH1750_SENSOR 2
#define LDR_SENSOR 3
#define AS7341_SENSOR 4

// Main AS7341L object
SparkFun_AS7341X as7341L;

// TSL2591 sensor object
TSL2591 tsl;

// BH1750 sensor object
BH1750 lightMeter;

// LDR sensor parameters
const int sensorPin = A0;

void setup() {
  DEV_ModuleInit();
  Serial.begin(9600);

  // Initialize the I2C bus
  Wire.begin();

  // Initialize AS7341L
  boolean as7341Result = as7341L.begin();

  // Initialize TSL2591
  tsl.begin();

  // Initialize BH1750
  lightMeter.begin();

  // If the board did not properly initialize print an error message and halt the system
  if (!as7341Result) {
    PrintErrorMessage();
    Serial.println("Check your connections. System halted !");
    while (true);
  }
}

void loop() {
  int sensorChoice = getUserInput();
  unsigned int channelReadings[12] = {0};

  switch (sensorChoice) {
    case TSL_SENSOR:
      readAndSaveTSLData();
      break;

    case BH1750_SENSOR:
      readAndSaveBH1750Data();
      break;

    case LDR_SENSOR:
      readAndSaveLDRData();
      break;

    case AS7341_SENSOR:
      readAndSaveAS7341Data(channelReadings);
      break;

    default:
      Serial.println("Invalid choice. Please select a valid sensor.");
      break;
  }

  delay(1000);
}

int getUserInput() {
  Serial.println("Select a sensor:");
  Serial.println("1. TSL2591 Ambient Light Sensor");
  Serial.println("2. BH1750 Ambient Light Sensor");
  Serial.println("3. LDR Sensor");
  Serial.println("4. AS7341L Spectral Sensor");

  while (!Serial.available()) {
    // Wait for user input
  }

  return Serial.parseInt();
}

void readAndSaveTSLData() {
  UWORD Lux = TSL2591_Read_Lux();

  Serial.print("Lux = ");
  Serial.println(Lux);
  
  // Save data to EEPROM
  EEPROM.write(0, Lux >> 8);
  EEPROM.write(1, Lux & 0xFF);
}

void readAndSaveBH1750Data() {
  float lux = lightMeter.readLightLevel();

  Serial.print("Light: ");
  Serial.print(lux);
  Serial.println(" lx");

  // Save data to EEPROM
  unsigned int luxInt = (unsigned int)lux;
  EEPROM.write(2, luxInt >> 8);
  EEPROM.write(3, luxInt & 0xFF);
}

void readAndSaveLDRData() {
  int sensorVal = analogRead(sensorPin);

  Serial.print("LDR Value: ");
  Serial.println(sensorVal);

  // Save data to EEPROM
  EEPROM.write(4, sensorVal >> 8);
  EEPROM.write(5, sensorVal & 0xFF);
}

void readAndSaveAS7341Data(unsigned int *channelReadings) {
  bool result = as7341L.readAllChannels(channelReadings);

  if (result) {
    // Save data to EEPROM
    for (int i = 0; i < 12; ++i) {
      EEPROM.write(i + 6, channelReadings[i] >> 8);
      EEPROM.write(i + 7, channelReadings[i] & 0xFF);
    }
  } else {
    PrintErrorMessage("Error");
  }
}

void PrintErrorMessage("Error") {
  // ... (same as in the provided AS7341 script)
}
