(ns blackfog.components.image
  (:require [reagent.core :as r]))

;; components/chat/image_upload.cljs
(def MAX_FILE_SIZE (* 1024 1024 5)) ;; 5MB 限制

(defn validate-file-size [file]
  (< (.-size file) MAX_FILE_SIZE))

(defn extract-base64-from-data-url [data-url]
  (let [base64-data (second (re-find #"^data:image/[^;]+;base64,(.+)$" data-url))]
    base64-data))

;; handlers/image.cljs
(defn upload-image-to-server [file]
  (-> (js/Promise.
       (fn [resolve reject]
         (let [form-data (js/FormData.)]
           (.append form-data "image" file)
           (-> (js/fetch "/api/upload"
                         #js {:method "POST"
                              :body form-data})
               (.then #(.json %))
               (.then #(resolve (.-url %)))
               (.catch reject)))))
      (.catch #(js/console.error "Upload failed:" %))))

(defn read-file-as-data-url [file]
  (js/Promise.
   (fn [resolve reject]
     (let [reader (js/FileReader.)]
       (set! (.-onload reader) #(resolve (.. % -target -result)))
       (set! (.-onerror reader) #(reject (.-error %)))
       (.readAsDataURL reader file)))))

(defn ImagePreview [{:keys [image-url on-remove]}]
  (r/with-let []
    [:div.image-preview.mt-2
     [:img.preview-image
      {:src image-url
       :alt "Preview"
       :style {:max-width "200px"
               :max-height "200px"}}]
     [:button.delete.ml-2
      {:on-click on-remove} "×"]]))

(defn ImageUploadButton [{:keys [on-file-select is-uploading]}]
  [:div.file.is-small.is-primary
   [:label.file-label
    [:input.file-input
     {:type "file"
      :accept "image/*"
      :disabled is-uploading
      :on-change #(when-let [file (-> % .-target .-files (aget 0))]
                    (on-file-select file))}]
    [:span.file-cta
     [:span.file-icon [:i.fas.fa-upload]]
     [:span.file-label
      (if is-uploading "处理中..." "选择图片")]]]])

(defn ImageUploadControl [{:keys [app-state]}]
  (r/with-let [is-uploading (r/atom false)
               preview-url (r/atom nil)
               img     (r/cursor app-state [:db :avatar :current-image])]
    (let [handle-file-select
          (fn [file]
            (if (validate-file-size file)
              (do
                (reset! is-uploading true)
                (-> (read-file-as-data-url file)
                    (.then (fn [data-url]
                             ;; 提取 base64 数据
                             (let [base64-data (extract-base64-from-data-url data-url)]
                               ;; 存储预览用的 data-url
                               (reset! preview-url data-url)
                               ;; 存储用于 Claude 的 base64 数据
                               (swap! app-state assoc-in
                                      [:db :avatar :current-image] base64-data))))
                    (.finally #(reset! is-uploading false))))
              (js/alert "文件大小超过限制")))

          handle-remove
          (fn []
            (reset! preview-url nil)
            (swap! app-state update-in
                   [:db :avatar :current-image] nil))]
      [:div.image-upload-control
       [ImageUploadButton
        {:on-file-select handle-file-select
         :is-uploading @is-uploading}]
       (when (and @preview-url @img)
         [ImagePreview
          {:image-url @preview-url
           :on-remove handle-remove}])])))
