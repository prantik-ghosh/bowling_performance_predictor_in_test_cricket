CREATE OR REPLACE FUNCTION update_bowler_home_adv_stat(v_year in int) RETURNS void AS $func$
/*
--
-- This function generates statistics regarding home advantage factor
-- for this specific player. It calculates a measure between 0 and 1
-- with a value of 0.5 indicating the player performs equally home or
-- away. A value closer to 1 indicates better home performance and
-- closer to 0 indiactes better away performance.
--
-- Note: This functionality could have been incorporated with
--       the procedure update_bowler_stat. But, it was not done
--       because we may have to modify this one further to include other
--       measures involving strike rate or average.
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

  v_mtchs_home         int;
  v_wkts_home          int;
  v_mtchs_away         int;
  v_wkts_away          int;
  v_bowl_home_adv      double precision;

BEGIN
  FOR cur_rec IN bowler_list LOOP
    SELECT
       count(1)
      ,sum(wkts)
    INTO
      v_mtchs_home
     ,v_wkts_home
    FROM
      bowling_data
    WHERE
          year < v_year
      and player = cur_rec.player
      and home_away = 1;

    SELECT
       count(1)
      ,sum(wkts)
    INTO
      v_mtchs_away
     ,v_wkts_away
    FROM
      bowling_data
    WHERE
          year < v_year
      and player = cur_rec.player
      and home_away = 0;

    IF (v_mtchs_home = 0 OR v_mtchs_away = 0) THEN
      v_bowl_home_adv = 0.5;
    ELSIF (v_wkts_home = 0 AND v_wkts_away = 0) THEN
      v_bowl_home_adv = 0.5;
    ELSIF (v_wkts_home = 0) THEN
      v_bowl_home_adv = 0;
    ELSIF (v_wkts_away = 0) THEN
      v_bowl_home_adv = 1;
    ELSE
      v_bowl_home_adv = (cast(v_wkts_home as float)/v_mtchs_home)/((cast(v_wkts_home as float)/v_mtchs_home) + (cast(v_wkts_away as float)/v_mtchs_away));
    END IF;

    UPDATE
       bowling_data
    SET
      bowl_home_adv = v_bowl_home_adv
    WHERE
          year = v_year
      and player = cur_rec.player
      and home_away = 1;

    UPDATE
       bowling_data
    SET
      bowl_home_adv = (1-v_bowl_home_adv)
    WHERE
          year = v_year
      and player = cur_rec.player
      and home_away = 0;

  END LOOP;
END;
$func$ LANGUAGE plpgsql;

