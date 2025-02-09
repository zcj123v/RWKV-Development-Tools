(ns blackfog.pages.index
  (:require [reagent.core :as r]
            [blackfog.components.chat-form :refer [Form]]))


(defn HomePage [{:keys [app-state
                        threads]}]
  (r/with-let []
    [Form app-state]))
