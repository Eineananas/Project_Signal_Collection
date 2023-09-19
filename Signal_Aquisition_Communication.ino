#include "SlowSoftWire.h"
#include "MMC5603NJ.h"
#include <WiFi.h>
#include "SparkFun_I2C_Mux_Arduino_Library.h" //Click here to get the library: http://librarymanager/All#SparkFun_I2C_Mux


// //----------------WiFi设置---------------------------
// // 设置 Wi-Fi SSID 和密码
// const char* ssid = "iWearSens";
// const char* password = "";
// // 设置服务器 IP 和端口号
// const char* server = "192.168.1.100";
// const uint16_t port = 8077;
// // 设置 UART2 的波特率
const uint32_t baudrate = 921600;
// // 设置每隔 N 微秒读取一次 UART2
// const uint32_t interval = 15000;
// // 定义 Wi-Fi 客户端对象和 UART2 对象 
// WiFiClient client;



// 定义TCA9548 初始I2C 地址
#define TCA9548_0_ADDR 0x70

//为每个IIC总线定义一个磁阻元件对象（端口已在MMC5603NJ.cpp中写死）
MMC5603NJ mmc0 = MMC5603NJ(21,22,0,MMC5603NJ_I2C_ADDR);
MMC5603NJ mmc1 = MMC5603NJ(33,32,1,MMC5603NJ_I2C_ADDR);
MMC5603NJ mmc2 = MMC5603NJ(25,26,2,MMC5603NJ_I2C_ADDR);

QWIICMUX mux[24]; //  实例化24个对象

short data[192][3];
short res[191][3];

void setup() {
  Serial.begin(baudrate);
  Serial.println();

  //初始化 TCA9548A
  for (int i=0;i<8;i++){
    mux[i].begin(TCA9548_0_ADDR+i,wire1);
    mux[i+8].begin(TCA9548_0_ADDR+i,wire2);
    mux[i+16].begin(TCA9548_0_ADDR+i,wire3);
  }
  
  //关闭所有 TCA9548的PORT
  for (int i=0;i<8;i++){
    mux[i].setPort(8);
    mux[i+8].setPort(8);
    mux[i+16].setPort(8);
  }
  //从IIC总线维度，分别初始化所有磁阻元件
    for (int i = 7; i >= 0; i--) {
      for (int j = 7; j>=0;j--){
          mux[i].setPort(j);
          mmc0.begin();
          mmc0.setContinuousMode(200);
          mmc0.setClockSpeed(I2C_CLOCK_400KHZ);
          mux[i].disablePort(j);
      }
    }
    for (int i = 0; i <= 7; i++) {
      for (int j = 0; j<=7;j++){
          mux[i].setPort(j);
          mmc0.begin();
          mmc0.setContinuousMode(200);
          mmc0.setClockSpeed(I2C_CLOCK_400KHZ);
          mux[i].disablePort(j);
      }
    }
    //IIC2
    for (int i = 7; i >= 0; i--) {
      for (int j = 7; j>=0;j--){
          mux[i+8].setPort(j);
          mmc1.begin();
          //mmc.softwareReset();
          mmc1.setContinuousMode(200);
          mmc1.setClockSpeed(I2C_CLOCK_400KHZ);
          mux[i+8].disablePort(j);
      }
    }
    for (int i = 0; i <= 7; i++) {
      for (int j = 0; j<=7;j++){
          mux[i+8].setPort(j);
          mmc1.begin();
          mmc1.setContinuousMode(200);
          mmc1.setClockSpeed(I2C_CLOCK_400KHZ);
          mux[i+8].disablePort(j);
      }
    }
    //IIC3
    for (int i = 7; i >= 0; i--) {
      for (int j = 7; j>=0;j--){
          mux[i+16].setPort(j);
          mmc2.begin();
          mmc2.setContinuousMode(200);
          mmc2.setClockSpeed(I2C_CLOCK_400KHZ);
          mux[i+16].disablePort(j);
      }
    }
    for (int i = 0; i <= 7; i++) {
      for (int j = 0; j<=7;j++){
          mux[i+16].setPort(j);
          mmc2.begin();
          mmc2.setContinuousMode(200);
          mmc2.setClockSpeed(I2C_CLOCK_400KHZ);
          mux[i+16].disablePort(j);
      }
    }

}

char buffer[200 * 3 * 4 + 100];
int offset = 0;
long startTime = 0;

long loopTime = 0;
long tmp = 0;

void loop() {
  loopTime = micros();
  for (int i = 0; i <= 7; i++) {
    for(int j=0; j<=7; j++){
        mux[i].setPort(j);
        mmc0.getMilliGauss(&data[i*8+7-j][0], &data[i*8+7-j][1], &data[i*8+7-j][2]);
        mux[i].setPort(8);
    }   
  }

  for (int i = 0; i <= 7; i++) {
    for(int j=0; j<=7; j++){
        mux[i+8].setPort(j);
        mmc1.getMilliGauss(&data[i*8+7-j+64][0], &data[i*8+7-j+64][1], &data[i*8+7-j+64][2]);
        mux[i+8].setPort(8);
    }   
  }

  for (int i = 0; i <= 7; i++) {
    for(int j=0; j<=7; j++){
        mux[i+16].setPort(j);
        mmc2.getMilliGauss(&data[i*8+7-j+128][0], &data[i*8+7-j+128][1], &data[i*8+7-j+128][2]);
        mux[i+16].setPort(8);
    }   
  }
  
//将二维数组拷贝到缓冲区中
  offset = 0;
    for (int i = 0; i < 192; i++) {
      if (((i>=32)&&(i<64))||((i>=128)&&(i<160))||(i>=176)){continue;}
        for (int j = 0; j < 3; j++) {
            offset += sprintf(buffer + offset, "%hd\t", data[i][j]);
        }
    }
    offset += sprintf(buffer + offset, "%ld\t\n", millis()-startTime);
    //offset += sprintf(buffer + offset, "\tR\n");

  //   //打印整个字符串缓冲区
  Serial.printf("%s", buffer);



  // 每隔 interval 微妙读取一次 
  // tmp = interval - (micros() - loopTime);
  // //Serial.println(tmp);
  // if(tmp > 0){
  //   delayMicroseconds(tmp);
  // }
  

}

void subarray(short (*data)[3], short (*res)[3],int i,int j) { 
    res[j][0] = data[i][0];
    res[j][1] = data[i][1];
    res[j][2] = data[i][2];
}
