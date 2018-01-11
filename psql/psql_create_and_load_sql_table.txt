DROP TABLE IF EXISTS
  bowling_data;

CREATE TABLE bowling_data(
   overs double precision
  ,mdns int
  ,runs int
  ,wkts int
  ,econ double precision
  ,ave double precision
  ,strike_rate double precision
  ,opposition varchar(30)
  ,ground varchar(50)
  ,home_away int
  ,bowling_arm int
  ,pace_spin int
  ,player varchar(50)
  ,country varchar(30)
  ,year int
);


COPY
  bowling_data
FROM
  '/home/prantik/galvanize/dsi-capstone-bowler-performance/data/bowling_data_raw_idx_n_header_deleted.csv'
WITH
  (FORMAT csv);


ALTER TABLE bowling_data
   ADD COLUMN balls int
  ,ADD COLUMN year1_mtchs_pld int
  ,ADD COLUMN year2_mtchs_pld int
  ,ADD COLUMN year3_mtchs_pld int
  ,ADD COLUMN year4_mtchs_pld int
  ,ADD COLUMN year5_mtchs_pld int
  ,ADD COLUMN year1_wkts_pm double precision
  ,ADD COLUMN year2_wkts_pm double precision
  ,ADD COLUMN year3_wkts_pm double precision
  ,ADD COLUMN year4_wkts_pm double precision
  ,ADD COLUMN year5_wkts_pm double precision
  ,ADD COLUMN bowler_agnst_oppo double precision
  ,ADD COLUMN oppo_agnst_bowl_typ double precision
  ,ADD COLUMN bowl_home_adv double precision
  ,ADD COLUMN ground_bowl_typ double precision
;


UPDATE
  bowling_data
SET
  balls = round(overs)*6 + (overs-round(overs))*10
;


/* All of the above are executed at the start before running the functions to create/manipulate the stats */
/* The below should be run at the end after the stat creation/manipulation is over */


COPY
  bowling_data
TO
  '/home/prantik/galvanize/dsi-capstone-bowler-performance/data/bowling_data_enhanced.csv'
WITH
  (FORMAT CSV, HEADER);

