(ns blackfog.components.dropdown
  (:require [reagent.core :as r]))

(defn DropDown [{:keys [options on-change selected-value placeholder]}]
  (let [show? (r/atom false)]
    (fn []
      [:div.dropdown {:class (when @show? "is-active")
                      :tab-index 0
                      :on-blur #(reset! show? false)}
       [:div.dropdown-trigger
        [:button.button
         {:aria-haspopup true
          :aria-controls "dropdown-menu"
          :on-click #(swap! show? not)}
         [:span (or (some #(when (= (:value %) selected-value) (:label %))
                          options)
                    placeholder
                    "Select an option")]
         [:span.icon.is-small
          [:i.fas.fa-angle-down {:aria-hidden true}]]]]

       [:div.dropdown-menu {:id "dropdown-menu" :role "menu"}
        [:div.dropdown-content
         (for [{:keys [value label disabled?]} options]
           ^{:key value}
           [:a.dropdown-item
            {:class (when (= value selected-value) "is-active")
             :on-click #(when-not disabled?
                          (println "========" )
                          (on-change value)
                          (reset! show? false))
             :disabled disabled?}
            label])]]])))
