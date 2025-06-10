FROM maven:3.9.3-eclipse-temurin-17 AS build
WORKDIR /app
COPY . .
RUN mvn clean package -DskipTests

# Use a minimal JDK image to run the app
FROM eclipse-temurin:17-jdk
WORKDIR /app
COPY --from=build /app/target/*.jar app.jar
COPY src/main/resources/keystore.p12 /app/keystore.p12
COPY best_model.onnx /app/best_model.onnx

# Spring Boot run on HTTPS
EXPOSE 8443
ENV SPRING_CONFIG_LOCATION=classpath:/application.properties
ENTRYPOINT ["java", "-jar", "app.jar"]
