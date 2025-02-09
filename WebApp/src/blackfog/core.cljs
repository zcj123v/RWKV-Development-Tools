(ns blackfog.core
  (:require [blackfog.dsl-protocol.impl]
            [reagent.core :as r]
            [datascript.core :as d]
            [cljs.reader :as reader]
            [reagent.dom.client :as rdom-client]
            [cljs.core.async :refer [go chan <! >! timeout put!]]
            [markdown.core :refer [md->html]]
            [blackfog.app-state.core :refer [app-state threads nexus]]
            [blackfog.dsl-protocol.core]
            [blackfog.persets.personas.core]
            [blackfog.nexus.core]
            [blackfog.pages.index :refer [HomePage]]))



;;; 上传组件
;;(defn app [] [index-page app-state])

(defn app [] [HomePage {:app-state app-state
                        :threads threads
                        :nexus nexus}])

(defn ^:export init []
  (let []
    (rdom-client/render (rdom-client/create-root
                         (js/document.getElementById "app")) [app])))
