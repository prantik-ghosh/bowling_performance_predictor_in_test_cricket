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

