(ns blackfog.dsl-protocol.core
  (:require [clojure.string :as str]
            [reagent.core :as r]
            [blackfog.app-state.core :refer [app-state]]
            [clojure.walk :refer [postwalk]]))

(defonce prompt-register! (r/atom {}))
