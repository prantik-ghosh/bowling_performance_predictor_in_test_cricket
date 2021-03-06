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




CREATE OR REPLACE FUNCTION update_bowler_oppo_stat(v_year in int) RETURNS void AS $func$
/*
--
-- This function generates bowler and opponent specific statistics.
-- It calculates the ratio of average wickets taken per match by
-- this player against this opponent vs against all opponents.
--
*/
DECLARE
  bowler_oppo_list CURSOR IS
    SELECT
      distinct player, opposition
    FROM
      bowling_data
    WHERE
      year = v_year;

  v_player             text;
  v_mtchs_tot          int;
  v_wkts_tot           int;
  v_mtchs_oppo         int;
  v_wkts_oppo          int;
  v_bowler_agnst_oppo  double precision;

BEGIN
  v_player := 'XYZ';

  FOR cur_rec IN bowler_oppo_list LOOP
    IF cur_rec.player != v_player THEN
      v_player := cur_rec.player;

      SELECT
         count(1)
        ,sum(wkts)
      INTO
         v_mtchs_tot
        ,v_wkts_tot
      FROM
        bowling_data
      WHERE
            year < v_year
        and player = v_player;
    END IF;

    SELECT
       count(1)
      ,sum(wkts)
    INTO
       v_mtchs_oppo
      ,v_wkts_oppo
    FROM
      bowling_data
    WHERE
          year < v_year
      and player = v_player
      and opposition = cur_rec.opposition;

    IF (v_mtchs_oppo = 0 OR v_mtchs_tot = 0 or v_wkts_tot = 0) THEN
      v_bowler_agnst_oppo = 1;
    ELSE
      v_bowler_agnst_oppo = (cast(v_wkts_oppo as float)/v_mtchs_oppo)/(cast(v_wkts_tot as float)/v_mtchs_tot);
    END IF;

    UPDATE
       bowling_data
    SET
      bowler_agnst_oppo = v_bowler_agnst_oppo
    WHERE
          year = v_year
      and player = v_player
      and opposition = cur_rec.opposition;

  END LOOP;
END;
$func$ LANGUAGE plpgsql;




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




CREATE OR REPLACE FUNCTION update_oppo_bowltyp_stat(v_year in int) RETURNS void AS $func$
/*
--
-- This function generates opponent and bowler type specific statistics.
-- It calculates the ratio of average wickets taken per match by bowler
-- of this type (pace/spin and right/left arm) against this opponent
-- vs bowler of all types against this opponent in the "last" 5 years.
--
*/
DECLARE
  oppo_bowltyp CURSOR IS
    SELECT
      distinct opposition, pace_spin, bowling_arm
    FROM
      bowling_data
    WHERE
      year = v_year
    ORDER BY
      1,2,3;

  v_opposition         text;
  v_mtchs_tot          int;
  v_wkts_tot           int;
  v_mtchs_btyp         int;
  v_wkts_btyp          int;
  v_oppo_agnst_btyp    double precision;

BEGIN
  v_opposition := 'XYZ';

  FOR cur_rec IN oppo_bowltyp LOOP
    IF cur_rec.opposition != v_opposition THEN
      v_opposition := cur_rec.opposition;

      SELECT
         count(1)
        ,sum(wkts)
      INTO
         v_mtchs_tot
        ,v_wkts_tot
      FROM
        bowling_data
      WHERE
            year < v_year
        and year >= v_year-5
        and opposition = v_opposition;
    END IF;

    SELECT
       count(1)
      ,sum(wkts)
    INTO
       v_mtchs_btyp
      ,v_wkts_btyp
    FROM
      bowling_data
    WHERE
          year < v_year
      and year >= v_year-5
      and opposition = v_opposition
      and pace_spin = cur_rec.pace_spin
      and bowling_arm = cur_rec.bowling_arm;

    IF (v_mtchs_btyp = 0 OR v_mtchs_tot = 0 or v_wkts_tot = 0) THEN
      v_oppo_agnst_btyp = 1;
    ELSE
      v_oppo_agnst_btyp = (cast(v_wkts_btyp as float)/v_mtchs_btyp)/(cast(v_wkts_tot as float)/v_mtchs_tot);
    END IF;

    UPDATE
       bowling_data
    SET
      oppo_agnst_bowl_typ = v_oppo_agnst_btyp
    WHERE
          year = v_year
      and opposition = v_opposition
      and pace_spin = cur_rec.pace_spin
      and bowling_arm = cur_rec.bowling_arm;

  END LOOP;
END;
$func$ LANGUAGE plpgsql;




CREATE OR REPLACE FUNCTION update_ground_bowltyp_stat(v_year in int) RETURNS void AS $func$
/*
--
-- This function generates ground and bowler type specific statistics.
-- It calculates the ratio of average wickets taken per match by bowler
-- of this type (pace/spin) in this ground vs average wickets taken per
-- match by bowler of all types in this ground.
--
*/
DECLARE
  ground_bowltyp CURSOR IS
    SELECT
      distinct ground, pace_spin
    FROM
      bowling_data
    WHERE
      year = v_year
    ORDER BY
      1,2;

  v_ground             text;
  v_mtchs_tot          int;
  v_wkts_tot           int;
  v_mtchs_btyp         int;
  v_wkts_btyp          int;
  v_ground_btyp        double precision;

BEGIN
  v_ground := 'XYZ';

  FOR cur_rec IN ground_bowltyp LOOP
    IF cur_rec.ground != v_ground THEN
      v_ground := cur_rec.ground;

      SELECT
         count(1)
        ,sum(wkts)
      INTO
         v_mtchs_tot
        ,v_wkts_tot
      FROM
        bowling_data
      WHERE
            year < v_year
        and ground = v_ground;
    END IF;

    SELECT
       count(1)
      ,sum(wkts)
    INTO
       v_mtchs_btyp
      ,v_wkts_btyp
    FROM
      bowling_data
    WHERE
          year < v_year
      and ground = v_ground
      and pace_spin = cur_rec.pace_spin;

    IF (v_mtchs_btyp = 0 OR v_mtchs_tot = 0 or v_wkts_tot = 0) THEN
      v_ground_btyp = 1;
    ELSE
      v_ground_btyp = (cast(v_wkts_btyp as float)/v_mtchs_btyp)/(cast(v_wkts_tot as float)/v_mtchs_tot);
    END IF;

    UPDATE
       bowling_data
    SET
      ground_bowl_typ = v_ground_btyp
    WHERE
          year = v_year
      and ground = v_ground
      and pace_spin = cur_rec.pace_spin;

  END LOOP;
END;
$func$ LANGUAGE plpgsql;

