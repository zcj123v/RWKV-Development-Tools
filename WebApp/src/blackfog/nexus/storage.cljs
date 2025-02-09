(ns blackfog.nexus.storage
  (:require [datascript.core :as d]))

(defprotocol DataBaseNexusPoint
  (transact [this tx-data] "添加")
  (query [this query-form] "查询")
  (find-by [this attr value] "匹配查询")
  (fulltext-search [this part] "全文匹配查询")

  )
