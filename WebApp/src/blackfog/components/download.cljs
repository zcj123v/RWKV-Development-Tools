(ns blackfog.components.download
  (:require [reagent.core :as r]
            ["react-dom/client" :as rdom]
            [cljs.reader :as reader]))


(defn download-data
  "将数据转换为文件并下载
   data: 要下载的数据
   filename: 文件名
   content-type: 文件类型（默认为 JSON）"
  [data filename & {:keys [content-type]
                    :or {content-type "application/json"}}]
  (let [blob (js/Blob. #js [(if (string? data)
                              data
                              (js/JSON.stringify (clj->js data) nil 2))]
                       #js {:type content-type})
        url (.createObjectURL js/URL blob)
        link (.createElement js/document "a")]
    (set! (.-href link) url)
    (set! (.-download link) filename)
    (.click link)
    (.revokeObjectURL js/URL url)))

(defn download-button
  "下载按钮组件
   data-atom: 包含要下载数据的 atom
   filename: 下载的文件名"
  [data filename f]
  [:button.button.small.is-small.is-primary
   {:on-click #(download-data (f data) filename)}
   "下载"])

(defn upload-button [on-upload]
  [:div.control
   [:label.button.is-small.is-primary
    {:style {:border-radius "8px"
             :transition "all 0.3s ease"
             :box-shadow "0 2px 4px rgba(0,0,0,0.1)"
             :&:hover {:transform "translateY(-1px)"
                       :box-shadow "0 4px 6px rgba(0,0,0,0.15)"}}}
    [:input.is-hidden
     {:type "file"
      :accept ".json"
      :style {:display "none"}
      :on-change (fn [e]
                   (let [file (-> e .-target .-files (aget 0))
                         reader (js/FileReader.)]
                     (set! (.-onload reader)
                           #(let [content (-> % .-target .-result)
                                  data (js->clj (js/JSON.parse content) :keywordize-keys true)]
                              (on-upload data)))
                     (.readAsText reader file)))}]
    [:span.icon.mr-2
     [:i.fas.fa-upload]]
    [:span "上传文件"]]])

;; (defn upload-button
;;   "文件上传按钮组件
;;    on-upload: 上传成功后的回调函数，接收解析后的数据作为参数"
;;   [on-upload]
;;   [:div.file.has-name.is-boxed
;;    [:label.file-label
;;     [:input.file-input
;;      {:type "file"
;;       :accept ".json"
;;       :on-change (fn [e]
;;                    (let [file (-> e .-target .-files (aget 0))
;;                          reader (js/FileReader.)]
;;                      (set! (.-onload reader)
;;                            #(let [content (-> % .-target .-result)
;;                                   data (js->clj (js/JSON.parse content) :keywordize-keys true)]
;;                               (on-upload data)))
;;                      (.readAsText reader file)))}]
;;     [:span.file-cta
;;      [:span.file-label "上传"]]]])


;; components/upload/image_upload.cljs
(defn ImageUploadButton [{:keys [on-upload]}]
  [:div.file.is-primary
   [:label.file-label
    [:input.file-input
     {:type "file"
      :accept "image/*"
      :on-change (fn [e] (when-let [file (-> e .-target .-files (aget 0))]
                           (let [reader (js/FileReader.)]
                             (set! (.-onload reader)
                                   #(on-upload (-> % .-target .-result)))
                             (.readAsDataURL reader file))))}]
    [:span.file-cta
     [:span.file-icon [:i.fas.fa-upload]]
     [:span.file-label "选择图片"]]]])

(defn ImagePreview [{:keys [image-url]}]
  (when image-url
    [:div.mt-2
     [:img {:src image-url
            :style {:max-width "200px"
                    :max-height "200px"}}]]))

(defn ImageUploader [{:keys [on-upload image-url]}]
  [:div
   [ImageUploadButton {:on-upload on-upload}]
   [ImagePreview {:image-url image-url}]])
