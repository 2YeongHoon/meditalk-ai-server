CREATE TABLE symptom_disease (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symptom VARCHAR(255) UNIQUE NOT NULL,
    disease VARCHAR(255) NOT NULL,
    department VARCHAR(255) NOT NULL
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

INSERT INTO symptom_disease (symptom, disease, department) VALUES 
('기침', '감기', '호흡기내과'),
('두통', '편두통', '신경과'),
('복통', '위염', '소화기내과');