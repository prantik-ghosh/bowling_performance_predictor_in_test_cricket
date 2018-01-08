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

