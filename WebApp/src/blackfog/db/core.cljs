(ns blackfog.db.core
  (:require [datascript.core :as d]
            [blackfog.persets.database :refer [database-init-data
                                               schema]]))
(defonce conn (d/create-conn schema))
