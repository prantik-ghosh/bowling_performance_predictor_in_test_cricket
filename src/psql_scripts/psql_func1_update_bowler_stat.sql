CREATE OR REPLACE FUNCTION update_bowler_stat(v_year in int) RETURNS void AS $func$
/*
--
-- This function generates bowler specific statistics (features) from
-- the existing data. It generates the following:
--   1. For each of the last 5 years how many matches this player has
--      played each year.
--   2. For each of the last 5 years how many wickets per match this
--      player has taken each year.
--
*/
DECLARE
  bowler_list CURSOR IS
    SELECT
      distinct player
    FROM
      bowling_data
    WHERE
      year = v_year;

  v_year1_mtchs_pld    int;
  v_year2_mtchs_pld    int;
  v_year3_mtchs_pld    int;
  v_year4_mtchs_pld    int;
  v_year5_mtchs_pld    int;
  v_year1_wkts_pm      double precision;
  v_year2_wkts_pm      double precision;
  v_year3_wkts_pm      double precision;
  v_year4_wkts_pm      double precision;
  v_year5_wkts_pm      double precision;
  v_wkts               int;

BEGIN
  FOR cur_rec IN bowler_list LOOP
    SELECT
       count(1)
      ,sum(wkts)
    INTO
      v_year1_mtchs_pld
     ,v_wkts
    FROM
      bowling_data
    WHERE
          year = v_year-1
      and player = cur_rec.player;

    IF v_year1_mtchs_pld = 0 THEN
      v_year1_wkts_pm = 0;
    ELSE
      v_year1_wkts_pm = cast(v_wkts as float)/v_year1_mtchs_pld;
    END IF;

    SELECT
       count(1)
      ,sum(wkts)
    INTO
      v_year2_mtchs_pld
     ,v_wkts
    FROM
      bowling_data
    WHERE
          year = v_year-2
      and player = cur_rec.player;

    IF v_year2_mtchs_pld = 0 THEN
      v_year2_wkts_pm = 0;
    ELSE
      v_year2_wkts_pm = cast(v_wkts as float)/v_year2_mtchs_pld;
    END IF;

    SELECT
       count(1)
      ,sum(wkts)
    INTO
      v_year3_mtchs_pld
     ,v_wkts
    FROM
      bowling_data
    WHERE
          year = v_year-3
      and player = cur_rec.player;

    IF v_year3_mtchs_pld = 0 THEN
      v_year3_wkts_pm = 0;
    ELSE
      v_year3_wkts_pm = cast(v_wkts as float)/v_year3_mtchs_pld;
    END IF;

    SELECT
       count(1)
      ,sum(wkts)
    INTO
      v_year4_mtchs_pld
     ,v_wkts
    FROM
      bowling_data
    WHERE
          year = v_year-4
      and player = cur_rec.player;

    IF v_year4_mtchs_pld = 0 THEN
      v_year4_wkts_pm = 0;
    ELSE
      v_year4_wkts_pm = cast(v_wkts as float)/v_year4_mtchs_pld;
    END IF;

    SELECT
       count(1)
      ,sum(wkts)
    INTO
      v_year5_mtchs_pld
     ,v_wkts
    FROM
      bowling_data
    WHERE
          year = v_year-5
      and player = cur_rec.player;

    IF v_year5_mtchs_pld = 0 THEN
      v_year5_wkts_pm = 0;
    ELSE
      v_year5_wkts_pm = cast(v_wkts as float)/v_year5_mtchs_pld;
    END IF;


    UPDATE
       bowling_data
    SET
       year1_mtchs_pld = v_year1_mtchs_pld
      ,year1_wkts_pm   = v_year1_wkts_pm
      ,year2_mtchs_pld = v_year2_mtchs_pld
      ,year2_wkts_pm   = v_year2_wkts_pm
      ,year3_mtchs_pld = v_year3_mtchs_pld
      ,year3_wkts_pm   = v_year3_wkts_pm
      ,year4_mtchs_pld = v_year4_mtchs_pld
      ,year4_wkts_pm   = v_year4_wkts_pm
      ,year5_mtchs_pld = v_year5_mtchs_pld
      ,year5_wkts_pm   = v_year5_wkts_pm
    WHERE
          year = v_year
      and player = cur_rec.player;

  END LOOP;
END;
$func$ LANGUAGE plpgsql;

