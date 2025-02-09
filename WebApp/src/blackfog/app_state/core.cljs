(ns blackfog.app-state.core
  (:require [reagent.core :as r]))


(defonce app-state
  (r/atom {:page {:index {:active :Eve}}
           :presence {:online {:Eve {:last-active 1732156800 ;; Unix时间戳

                                     :status :typing}

                               :LinXia {:last-active 1732156815

                                        :status :online}

                               :bonsai {:last-active 1732156700

                                        :status :offline}}}}))


(defonce threads  (r/atom {}))

(defonce personas (r/atom {:Eve {:status ""
                                 :mood ""}}))

(defonce nexus
  (r/atom {:Eve    {:services :claude
                    :history []}
           :LinXia {:services :deepseek
                    :history []}
           :Echo   {:services :extractor
                    :history []}}))
