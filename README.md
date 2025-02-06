**VGG19 tabanlı bir U-Net modeli** kullanarak **tıbbi veya genel görüntü segmentasyonu** gerçekleştirmektedir. İşleyişi aşağıdaki adımlardan oluşmaktadır:  

1. **Veri Yükleme ve Ön İşleme:**  
   - Google Drive'dan **görüntü ve maske verileri** yüklenir.  
   - Görseller **512x512 boyutuna** yeniden ölçeklendirilir ve renk kanalları düzenlenir.  

2. **Eğitim ve Test Verilerinin Hazırlanması:**  
   - Görseller ve karşılık gelen maskeler **numpy dizilerine** dönüştürülür.  
   - **%80 eğitim, %20 test** olacak şekilde veri ayrımı yapılır.  

3. **VGG19 Tabanlı U-Net Modelinin Oluşturulması:**  
   - Önceden eğitilmiş **VGG19 modeli** kullanılarak bir **U-Net segmentasyon ağı** oluşturulur.  
   - Model, **çift konvolüsyon** ve **kodlayıcı-çözümleyici (encoder-decoder) blokları** ile yapılandırılır.  

4. **Modelin Eğitilmesi:**  
   - Model, **Adam optimizasyon algoritması** ve **binary crossentropy kaybı** ile eğitilir.  
   - **Eğitim doğruluğu ve kayıp değerleri** görselleştirilir.  

5. **Modelin Değerlendirilmesi ve Performans Analizi:**  
   - Model test verileri üzerinde tahminler yapar.  
   - **Jaccard (IoU) skoru** ile modelin performansı değerlendirilir.  
   - **Farklı eşik değerleri** kullanılarak segmentasyon sonuçları karşılaştırılır.  

6. **Sonuçların Görselleştirilmesi:**  
   - **Orijinal görüntü, model tahmini ve gerçek maske** yan yana gösterilir.  
   - **IoU skorları grafiğe dökülerek** farklı eşik değerlerindeki performans analiz edilir.  
